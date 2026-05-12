// clang-format off
/*----------------------------------------------------------------------
  PuReMD - Purdue ReaxFFSVE Molecular Dynamics Program

  Copyright (2010) Purdue University
  Hasan Metin Aktulga, hmaktulga@lbl.gov
  Joseph Fogarty, jcfogart@mail.usf.edu
  Sagar Pandit, pandit@usf.edu
  Ananth Y Grama, ayg@cs.purdue.edu

  Please cite the related publication:
  H. M. Aktulga, J. C. Fogarty, S. A. Pandit, A. Y. Grama,
  "Parallel Reactive Molecular Dynamics: Numerical Methods and
  Algorithmic Techniques", Parallel Computing, 38 (4-5), 245-259.

  This program is free software; you can redistribute it and/or
  modify it under the terms of the GNU General Public License as
  published by the Free Software Foundation; either version 2 of
  the License, or (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
  See the GNU General Public License for more details:
  <https://www.gnu.org/licenses/>.
  ----------------------------------------------------------------------*/

#include "reaxff_api_sve.h"

#include "pair.h"
#include "neigh_list.h"
#include "atom.h"
#include "error.h"

#include "cluster_neigh.h"
#include <arm_sve.h>
#include "sve_util.h"
#include <omp.h>
#include "thr_sve.h"

#include <cmath>

#define CLUSTERSIZE 8

using namespace LAMMPS_NS;

namespace ReaxFFSVE {
  void Compute_Polarization_Energy(reax_system *system, simulation_data *data, storage *workspace)
  {
    data->my_en.e_pol = 0.0;
    double local_e_pol = 0.0;

    #pragma omp parallel for schedule(dynamic, 64) reduction(+:local_e_pol)
    for (int ii = 0; ii < system->n; ii++) 
    {
      int type_i = system->map[system->type[ii]];
      if (type_i < 0) continue;
      double qq = system->q[ii];
      double en_tmp = KCALpMOL_to_EV * (system->reax_param.sbp[type_i].chi * qq +
                              (system->reax_param.sbp[type_i].eta * 0.5) * SQR(qq));
      local_e_pol += en_tmp;
    }
    data->my_en.e_pol += local_e_pol;
    if (system->pair_ptr->eflag_either)
    {
      system->pair_ptr->eng_vdwl += local_e_pol;
    }
  }

  void vdW_Coulomb_Energy(reax_system *system, control_params *control,
                          simulation_data *data, storage *workspace,
                          reax_list **lists)
  {
    const double SMALL = 0.0001;
    int natoms = system->list->inum;

    using NeighEnt = ClusterNeighEntry<1>;

    //using list from LAMMPS:
    int *ilist, *jlist, *numneigh, **firstneigh;
    ilist       = system->list->ilist;

    int *numneigh_cluster = system->list->numneigh_inner;
    NeighEnt **firstneigh_cluster = (NeighEnt **) system->list->firstneigh_inner;
    svuint64_t svmaskbits = svlsl_m(svptrue_b64(), svdup_u64(1), svindex_u64(0, 1));

    //numneigh    = system->list->numneigh;
    //firstneigh  = system->list->firstneigh;
    double (*x)[3]  = (double(*)[3])system->x[0];
    double (*f)[3]  = (double(*)[3])workspace->f[0];
    double *q   = system->q;
    int *type   = system->type;
    int ntypes = system->ntypes;
    int *map    = system->map; 
    double xtmp, ytmp, ztmp;
    double del[3], rsq;
    int jnum, itype, jtype;
    
    int num_clusters = (natoms + CLUSTERSIZE - 1) / CLUSTERSIZE;

    svd_t one_third = svdup_f64(0.33333333333333);
    svd_t half_one = svdup_f64(0.5);
    svd_t one = svdup_f64(1.0);
    svd_t two = svdup_f64(2.0);

    svd_t C_ele0 = svdup_f64(C_ele);
    svd_t small_vec = svdup_f64(SMALL);
    svd_t cutijsq0 = svdup_f64(control->nonb_cut * control->nonb_cut);

    int nthr = omp_get_max_threads();
    double local_e_vdW = 0.0;
    double local_e_ele = 0.0;
    double local_e_vdwl = 0.0;
    double local_e_coul = 0.0;

  #pragma omp parallel num_threads(nthr) reduction(+:local_e_vdW, local_e_ele, local_e_vdwl, local_e_coul)
  {
    int i, j, pj;
    int start_i, end_i;
    rc_tagint orig_i, orig_j;
    double powr_vdW1, powgi_vdW1;
    double tmp, r_ij, fn13, exp1, exp2;
    double Tap, dTap, dfn13, CEvd, CEclmb, de_core;
    double dr3gamij_1, dr3gamij_3;
    double e_ele;
    double r_ij5, r_ij6, re6;
    double bond_softness, d_bond_softness, d, effpot_diff;
    two_body_parameters *twbp;

    double pe_vdw, f_tmp, delij[3];

    double p_vdW1 = system->reax_param.gp.l[28];
    svd_t p_vdW1_vec = svdup_f64(system->reax_param.gp.l[28]);
    svd_t p_vdW1_sub_two_vec = svdup_f64(system->reax_param.gp.l[28] - 2.0);

    double p_vdW1i = 1.0 / p_vdW1;
    svd_t p_vdW1i_vec = svdup_f64(p_vdW1i);
    svd_t p_vdW1i_sub_one_vec = svdup_f64(p_vdW1i - 1.0);

    double e_core = 0;
    svd_t e_core0 = svdup_f64(0.0);

    double e_vdW = 0;
    svd_t e_vdW0 = svdup_f64(0.0);

    double e_lg  = 0.0;
    double de_lg = 0.0;
    svd_t e_lg0 = svdup_f64(0.0);

    int ithr = omp_get_thread_num();
    double (*my_f)[3] = (double(*)[3])workspace->forceTmp[ithr][0];

    #pragma omp for schedule(dynamic, 64)
    for (int ic = 0; ic < num_clusters; ic++) {
      int iis = ic * CLUSTERSIZE;
      int iie = std::min(iis+CLUSTERSIZE, natoms);
      int iicnt = iie - iis;

      svbool_t predi32 = svwhilelt_b32(0, iicnt);
      svbool_t predi64 = svwhilelt_b64(0, iicnt);

      svint32_t is32 = svld1(predi32, ilist + iis);
      svint64_t is64 = svunpklo_s64(is32);

      svint32_t itype_before = svld1_gather_offset(predi32, type, is32*4);
      svint64_t itypei = svunpklo_s64(svld1_gather_offset(predi32, map, itype_before*4));
      
      #ifdef LAMMPS_BIGBIG
      svint64_t itag = svld1_gather_index(predi64, system->tag, is64);
      #else
      svint64_t itag = svunpklo_s64(svld1_gather_index(predi32, system->tag, is32));
      #endif

      svd_t xi = svld1_gather_offset(predi64, (double*)x, is64*24);
      svd_t yi = svld1_gather_offset(predi64, (double*)x, is64*24+8);
      svd_t zi = svld1_gather_offset(predi64, (double*)x, is64*24+16);
      svd_t qi = svld1_gather_index(predi64, q, is64);

      int itypes[CLUSTERSIZE];
      svst1_s32(predi32, itypes, itype_before);

      double gamma_w_arr[ntypes + 1][iicnt];
      double alpha_arr[ntypes + 1][iicnt];
      double r_vdW_arr[ntypes + 1][iicnt];
      double D_arr[ntypes + 1][iicnt];
      double gamma_arr[ntypes + 1][iicnt];

      svd_t gamma_w0[ntypes + 1];
      svd_t alpha0[ntypes + 1];
      svd_t r_vdW0[ntypes + 1];
      svd_t D0[ntypes + 1];
      svd_t gamma0[ntypes + 1];

      // added by szh for inner wall:
      double ecore0_arr[ntypes + 1][iicnt];
      double acore0_arr[ntypes + 1][iicnt];
      double rcore0_arr[ntypes + 1][iicnt];
  
      svd_t ecore0[ntypes + 1];
      svd_t acore0[ntypes + 1];
      svd_t rcore0[ntypes + 1];
      // end

      for(int jtemp=1; jtemp<=ntypes; jtemp++) {
        for(int itemp=0; itemp<iicnt; itemp++) {
          two_body_parameters *twbp0 = &(system->reax_param.tbp[map[itypes[itemp]]][map[jtemp]]);	
          gamma_w_arr[jtemp][itemp] = 1.0 / twbp0->gamma_w; 
          alpha_arr[jtemp][itemp] = twbp0->alpha; 
          r_vdW_arr[jtemp][itemp] = 1.0 / twbp0->r_vdW; 
          D_arr[jtemp][itemp] = twbp0->D; 
          gamma_arr[jtemp][itemp] = twbp0->gamma; 
          // added by szh for inner wall:
          ecore0_arr[jtemp][itemp] = twbp0->ecore; 
          acore0_arr[jtemp][itemp] = twbp0->acore; 
          rcore0_arr[jtemp][itemp] = 1.0 / twbp0->rcore; 
          // end
	      }
        gamma_w0[jtemp] = svld1(predi64, &gamma_w_arr[jtemp][0]);
        alpha0[jtemp] = svld1(predi64, &alpha_arr[jtemp][0]);
        r_vdW0[jtemp] = svld1(predi64, &r_vdW_arr[jtemp][0]);
        D0[jtemp] = svld1(predi64, &D_arr[jtemp][0]);
        gamma0[jtemp] = svld1(predi64, &gamma_arr[jtemp][0]);
        // added by szh for inner wall:
        ecore0[jtemp] = svld1(predi64, &ecore0_arr[jtemp][0]);
        acore0[jtemp] = svld1(predi64, &acore0_arr[jtemp][0]);
        rcore0[jtemp] = svld1(predi64, &rcore0_arr[jtemp][0]);
        // end
      } 

      NeighEnt *jlist_cluster = firstneigh_cluster[ic];
      int jnum_cluster = numneigh_cluster[ic];

      svd_t fxi = svdup_f64(0), fyi = svdup_f64(0), fzi = svdup_f64(0);
      for(int jj = 0; jj < jnum_cluster; jj++) {
        NeighEnt jent0 = jlist_cluster[jj];
        int j0 = jent0.j;
        j0 &= NEIGHMASK;
	

        int jtype0_nomap = type[j0];
        int jtype0 = map[jtype0_nomap];

        svd_t xj0 = svdup_f64(x[j0][0]);
        svd_t yj0 = svdup_f64(x[j0][1]);
        svd_t zj0 = svdup_f64(x[j0][2]);

	      svint64_t js64 = svdup_s64(j0);

	      svbool_t newton0 = predi64;
	
        svbool_t j_gt_i0 = svcmpgt(predi64, js64, is64);
        if (!svptest_any(svptrue_b64(), j_gt_i0)) continue;

        if(j0 < natoms) {
                newton0 = j_gt_i0;
        } else {
          svint64_t jtag0 = svunpklo_s64(svdup_s32(system->tag[j0]));
          // 1
          svbool_t orig_i_lt_j0 = svcmplt(j_gt_i0, itag, jtag0);
          // 2
          svbool_t orig_i_eq_j0 = svcmpeq(j_gt_i0, itag, jtag0);

          svd_t z_diff0 = zj0 - zi;
          // 2_1
          svbool_t z_gt_small0 = svcmpgt(orig_i_eq_j0, z_diff0, small_vec);
          // 2_2
          svbool_t z_lt_small0 = svcmplt(orig_i_eq_j0, z_diff0, small_vec);
          svd_t y_diff0 = yj0 - yi;
          svbool_t y_gt_small0 = svcmpgt(z_lt_small0, y_diff0, small_vec); 

          newton0 = svorr_z(j_gt_i0, orig_i_lt_j0, svorr_z(orig_i_eq_j0, z_gt_small0, y_gt_small0));

	        if (!svptest_any(svptrue_b64(), newton0)) continue;
	      }


        svd_t delx0 = xj0 - xi;
        svd_t dely0 = yj0 - yi;
        svd_t delz0 = zj0 - zi;
	
        svd_t rsq0 = delx0 * delx0 + dely0 * dely0 + delz0 * delz0;

        svbool_t incut0 = svcmple(newton0, rsq0, cutijsq0);

        if (!svptest_any(svptrue_b64(), incut0)) continue;

        svd_t r_ij0 = svsqrt_f64_z(incut0, rsq0);
	
        svd_t Tap0 = svdup_f64(workspace->Tap[7]);
        for(int i=6; i>=0; i--) {
          Tap0 = svmad_f64_z(incut0, Tap0, r_ij0, svdup_f64(workspace->Tap[i]));
        }

        svd_t dTap0 = svdup_f64(7.0 * workspace->Tap[7]);
        dTap0 = svmad_f64_z(incut0, dTap0, r_ij0, svdup_f64(6.0 * workspace->Tap[6]));
        dTap0 = svmad_f64_z(incut0, dTap0, r_ij0, svdup_f64(5.0 * workspace->Tap[5]));
        dTap0 = svmad_f64_z(incut0, dTap0, r_ij0, svdup_f64(4.0 * workspace->Tap[4]));
        dTap0 = svmad_f64_z(incut0, dTap0, r_ij0, svdup_f64(3.0 * workspace->Tap[3]));
        dTap0 = svmad_f64_z(incut0, dTap0, r_ij0, svdup_f64(2.0 * workspace->Tap[2]));
        svd_t workspaceTap1 = svdup_f64(workspace->Tap[1]);
        dTap0 += workspaceTap1 / r_ij0;

        svd_t CEvd0;
        if (system->reax_param.gp.vdw_type==1 || system->reax_param.gp.vdw_type==3) {
          // shielding
          svd_t r_ij0__and_gamma_w_log_in[2] = {r_ij0, gamma_w0[jtype0_nomap]};
          svd_t r_ij0__and_gamma_w_log_out[2];
          svnxp_log<2, 4>(r_ij0__and_gamma_w_log_out, r_ij0__and_gamma_w_log_in);
          svd_t p_and_pi_vdW1_vec_exp_in[2] = {p_vdW1_vec * r_ij0__and_gamma_w_log_out[0], p_vdW1_vec * r_ij0__and_gamma_w_log_out[1]};
          svd_t p_and_pi_vdW1_vec_exp_out[2];
          svnxp_exp<2, 4>(p_and_pi_vdW1_vec_exp_out, p_and_pi_vdW1_vec_exp_in);
          svd_t powr_powgi_vdW1_log_in[1] = {p_and_pi_vdW1_vec_exp_out[0] + p_and_pi_vdW1_vec_exp_out[1]};
          svd_t powr_powgi_vdW1_log_out[1];
          svnxp_log<1, 4>(powr_powgi_vdW1_log_out, powr_powgi_vdW1_log_in);
          svd_t fn13_exp_in[1] = {p_vdW1i_vec * powr_powgi_vdW1_log_out[0]};
          svd_t fn13_exp_out[1];
          svnxp_exp<1, 4>(fn13_exp_out, fn13_exp_in);
          svd_t exp_arg0 = alpha0[jtype0_nomap] * (one - fn13_exp_out[0] * r_vdW0[jtype0_nomap]);
          svd_t exp1_and_2_exp_And_dfn13_1_0_and_1_in[4] = {exp_arg0, half_one * exp_arg0, p_vdW1i_sub_one_vec * powr_powgi_vdW1_log_out[0], p_vdW1_sub_two_vec * r_ij0__and_gamma_w_log_out[0]};
          svd_t exp1_and_2_exp_And_dfn13_1_0_and_1_out[4];
          svnxp_exp<4, 4>(exp1_and_2_exp_And_dfn13_1_0_and_1_out, exp1_and_2_exp_And_dfn13_1_0_and_1_in);
          e_vdW0 = D0[jtype0_nomap] * (exp1_and_2_exp_And_dfn13_1_0_and_1_out[0] - two * exp1_and_2_exp_And_dfn13_1_0_and_1_out[1]);
          // data->my_en.e_vdW += svaddv_f64(incut0, Tap0 * e_vdW0);
          // eng_virial[0] += svaddv_f64(incut0, Tap0 * e_vdW0);
          local_e_vdW += svaddv_f64(incut0, Tap0 * e_vdW0);
          svd_t dfn13_vec0 = exp1_and_2_exp_And_dfn13_1_0_and_1_out[2] * exp1_and_2_exp_And_dfn13_1_0_and_1_out[3];
          CEvd0 = dTap0 * e_vdW0 - Tap0 * D0[jtype0_nomap] * (alpha0[jtype0_nomap] * r_vdW0[jtype0_nomap]) * (exp1_and_2_exp_And_dfn13_1_0_and_1_out[0] - exp1_and_2_exp_And_dfn13_1_0_and_1_out[1]) * dfn13_vec0;  
        } else {
          // no shielding
          // this part has not been tested, and its correctness is uncertain
          svd_t exp_arg0 = alpha0[jtype0_nomap] * (one - r_ij0 * r_vdW0[jtype0_nomap]);
          svd_t exp1_and_2_in[2] = {exp_arg0, half_one * exp_arg0};
          svd_t exp1_and_2_out[2];
          svnxp_exp<2, 4>(exp1_and_2_out, exp1_and_2_in);
          e_vdW0 = D0[jtype0_nomap] * (exp1_and_2_out[0] - two * exp1_and_2_out[1]);
          // data->my_en.e_vdW += svaddv_f64(incut0, Tap0 * e_vdW0);
          // eng_virial[0] += svaddv_f64(incut0, Tap0 * e_vdW0);
          local_e_vdW += svaddv_f64(incut0, Tap0 * e_vdW0);
          CEvd0 = dTap0 * e_vdW0 - Tap0 * D0[jtype0_nomap] * (alpha0[jtype0_nomap] * r_vdW0[jtype0_nomap]) * (exp1_and_2_out[0] - exp1_and_2_out[1]) / r_ij0;
        }

        if (system->reax_param.gp.vdw_type==2 || system->reax_param.gp.vdw_type==3) {
          // inner wall
          svd_t exp_arg0 = acore0[jtype0_nomap] * (one - r_ij0 * rcore0[jtype0_nomap]);
          svd_t exp_in[1] = {exp_arg0};
          svd_t exp_out[1];
          svnxp_exp<1, 4>(exp_out, exp_in);
          e_core0 = ecore0[jtype0_nomap] * exp_out[0];
          // data->my_en.e_vdW += svaddv_f64(incut0, Tap0 * e_core0);
          // eng_virial[0] += svaddv_f64(incut0, Tap0 * e_core0);
          local_e_vdW += svaddv_f64(incut0, Tap0 * e_core0);
          svd_t de_core0 = -(acore0[jtype0_nomap] * rcore0[jtype0_nomap]) * e_core0;
          CEvd0 += dTap0 * e_core0 + Tap0 * de_core0 / r_ij0;
          // lg correction, only if lgvdw is yes
          if (control->lgflag) {
            system->error_ptr->all(FLERR, "LG correction with inner wall is not implemented yet.");
          }
        }

        svd_t dr3gamij_1_vec0 = r_ij0 * r_ij0 * r_ij0 + gamma0[jtype0_nomap];
        svd_t dr3gamij_3_log_in[1] = {dr3gamij_1_vec0};
        svd_t dr3gamij_3_log_out[1];
        svnxp_log<1, 4>(dr3gamij_3_log_out, dr3gamij_3_log_in);
        svd_t dr3gamij_3_exp_in[1] = {one_third * dr3gamij_3_log_out[0]};
        svd_t dr3gamij_3_exp_out[1];
        svnxp_exp<1, 4>(dr3gamij_3_exp_out, dr3gamij_3_exp_in);
        svd_t dr3gamij_3_vec0 = dr3gamij_3_exp_out[0];

        svd_t tmp0 = Tap0 / dr3gamij_3_vec0;

        svd_t qj0 = svdup_f64(q[j0]);
        svd_t e_ele0 = C_ele0 * qi * qj0 * tmp0;
        // data->my_en.e_ele += svaddv_f64(incut0, e_ele0);
        // eng_virial[1] += svaddv_f64(incut0, e_ele0);
        local_e_ele += svaddv_f64(incut0, e_ele0);

        svd_t CEclmb0 = C_ele0 * qi * qj0 * (dTap0 - Tap0 * r_ij0 / dr3gamij_1_vec0) / dr3gamij_3_vec0;

	      if (system->pair_ptr->evflag){
          // system->pair_ptr->eng_vdwl += svaddv_f64(incut0, Tap0 * (e_vdW0 + e_core0 + e_lg0)); 
          // system->pair_ptr->eng_coul += svaddv_f64(incut0, e_ele0); 
          // eng_virial[2] += svaddv_f64(incut0, Tap0 * (e_vdW0 + e_core0 + e_lg0));
          // eng_virial[3] += svaddv_f64(incut0, e_ele0);
          local_e_vdwl += svaddv_f64(incut0, Tap0 * (e_vdW0 + e_core0 + e_lg0));
          local_e_coul += svaddv_f64(incut0, e_ele0);
        }

        svd_t tempx0 = -(CEvd0 + CEclmb0) * delx0;
        svd_t tempy0 = -(CEvd0 + CEclmb0) * dely0;
        svd_t tempz0 = -(CEvd0 + CEclmb0) * delz0;

        fxi = svadd_m(incut0, fxi, tempx0);
        fyi = svadd_m(incut0, fyi, tempy0);
        fzi = svadd_m(incut0, fzi, tempz0);

        my_f[j0][0] -= svaddv_f64(incut0, tempx0); 
        my_f[j0][1] -= svaddv_f64(incut0, tempy0); 
        my_f[j0][2] -= svaddv_f64(incut0, tempz0); 
      }

      svd_t cfxi = svld1_gather_offset(predi64, (double*)my_f, is64*24);
      svd_t cfyi = svld1_gather_offset(predi64, (double*)my_f, is64*24+8);
      svd_t cfzi = svld1_gather_offset(predi64, (double*)my_f, is64*24+16);

      svst1_scatter_offset(predi64, (double*)my_f, is64 * 24, cfxi + fxi);
      svst1_scatter_offset(predi64, (double*)my_f, is64 * 24 + 8, cfyi + fyi);
      svst1_scatter_offset(predi64, (double*)my_f, is64 * 24 + 16, cfzi + fzi);
    }
  } //end OMP()
    data->my_en.e_vdW += local_e_vdW;
    data->my_en.e_ele += local_e_ele;
    if (system->pair_ptr->eflag_either) {
      system->pair_ptr->eng_vdwl += local_e_vdwl;
      system->pair_ptr->eng_coul += local_e_coul;
    }

    Compute_Polarization_Energy( system, data, workspace );  
  }

  void Tabulated_vdW_Coulomb_Energy(reax_system *system, control_params *control,
                                     simulation_data *data, storage *workspace,
                                     reax_list **lists)
  {
    int i, j, pj, r, natoms;
    int type_i, type_j, tmin, tmax;
    int start_i, end_i, flag;
    rc_tagint orig_i, orig_j;
    double r_ij, base, dif;
    double e_vdW, e_ele;
    double CEvd, CEclmb, SMALL = 0.0001;
    double f_tmp, delij[3];
    double bond_softness, d_bond_softness, d, effpot_diff;

    far_neighbor_data *nbr_pj;
    reax_list *far_nbrs;
    LR_lookup_table *t;
    LR_lookup_table ** & LR = system->LR;

    natoms = system->n;
    far_nbrs = (*lists) + FAR_NBRS;

    e_ele = e_vdW = 0;

    for (i = 0; i < natoms; ++i) {
      type_i  = system->my_atoms[i].type;
      if (type_i < 0) continue;
      start_i = Start_Index(i,far_nbrs);
      end_i   = End_Index(i,far_nbrs);
      orig_i  = system->my_atoms[i].orig_id;

      for (pj = start_i; pj < end_i; ++pj) {
        nbr_pj = &(far_nbrs->select.far_nbr_list[pj]);
        j = nbr_pj->nbr;
        type_j = system->my_atoms[j].type;
        if (type_j < 0) continue;
        orig_j  = system->my_atoms[j].orig_id;

        flag = 0;
        if (nbr_pj->d <= control->nonb_cut) {
          if (j < natoms) flag = 1;
          else if (orig_i < orig_j) flag = 1;
          else if (orig_i == orig_j) {
            if (nbr_pj->dvec[2] > SMALL) flag = 1;
            else if (fabs(nbr_pj->dvec[2]) < SMALL) {
              if (nbr_pj->dvec[1] > SMALL) flag = 1;
              else if (fabs(nbr_pj->dvec[1]) < SMALL && nbr_pj->dvec[0] > SMALL)
                flag = 1;
            }
          }
        }

        if (flag) {

          r_ij   = nbr_pj->d;
          tmin  = MIN(type_i, type_j);
          tmax  = MAX(type_i, type_j);
          t = &(LR[tmin][tmax]);

          /* Cubic Spline Interpolation */
          r = (int)(r_ij * t->inv_dx);
          if (r == 0)  ++r;
          base = (double)(r+1) * t->dx;
          dif = r_ij - base;

          e_vdW = ((t->vdW[r].d*dif + t->vdW[r].c)*dif + t->vdW[r].b)*dif +
            t->vdW[r].a;

          e_ele = ((t->ele[r].d*dif + t->ele[r].c)*dif + t->ele[r].b)*dif +
            t->ele[r].a;
          e_ele *= system->my_atoms[i].q * system->my_atoms[j].q;

          data->my_en.e_vdW += e_vdW;
          data->my_en.e_ele += e_ele;

          CEvd = ((t->CEvd[r].d*dif + t->CEvd[r].c)*dif + t->CEvd[r].b)*dif +
            t->CEvd[r].a;

          CEclmb = ((t->CEclmb[r].d*dif+t->CEclmb[r].c)*dif+t->CEclmb[r].b)*dif +
            t->CEclmb[r].a;
          CEclmb *= system->my_atoms[i].q * system->my_atoms[j].q;

          /* tally into per-atom energy */
          if (system->pair_ptr->evflag) {
            rvec_ScaledSum(delij, 1., system->my_atoms[i].x,
                            -1., system->my_atoms[j].x);
            f_tmp = -(CEvd + CEclmb);
            system->pair_ptr->ev_tally(i,j,natoms,1,e_vdW,e_ele,
                                       f_tmp,delij[0],delij[1],delij[2]);
          }

          rvec_ScaledAdd(workspace->f[i], -(CEvd + CEclmb), nbr_pj->dvec);
          rvec_ScaledAdd(workspace->f[j], +(CEvd + CEclmb), nbr_pj->dvec);
        }
      }
    }

    /* contribution to energy and gradients (atoms and cell)
     * due to geometry-dependent terms in the ACKS2
     * kinetic energy */
    if (system->acks2_flag)
    for( i = 0; i < natoms; ++i ) {
      if (system->my_atoms[i].type < 0) continue;
      start_i = Start_Index(i, far_nbrs);
      end_i   = End_Index(i, far_nbrs);
      orig_i  = system->my_atoms[i].orig_id;

      for( pj = start_i; pj < end_i; ++pj ) {
        nbr_pj = &(far_nbrs->select.far_nbr_list[pj]);
        j = nbr_pj->nbr;
        if (system->my_atoms[j].type < 0) continue;
        orig_j  = system->my_atoms[j].orig_id;

        flag = 0;

        /* kinetic energy terms */
        double xcut = 0.5 * (system->reax_param.sbp[ system->my_atoms[i].type ].bcut_acks2
                             + system->reax_param.sbp[ system->my_atoms[j].type ].bcut_acks2);

        if(nbr_pj->d <= xcut) {
          if (j < natoms) flag = 1;
          else if (orig_i < orig_j) flag = 1;
          else if (orig_i == orig_j) {
            if (nbr_pj->dvec[2] > SMALL) flag = 1;
            else if (fabs(nbr_pj->dvec[2]) < SMALL) {
              if (nbr_pj->dvec[1] > SMALL) flag = 1;
              else if (fabs(nbr_pj->dvec[1]) < SMALL && nbr_pj->dvec[0] > SMALL)
                flag = 1;
            }
          }
        }

        if (flag) {

          d = nbr_pj->d / xcut;
          bond_softness = system->reax_param.gp.l[34] * pow( d, 3.0 )
                        * pow( 1.0 - d, 6.0 );

          if ( bond_softness > 0.0 )
          {
            /* Coulombic energy contribution */
            effpot_diff = workspace->s[system->N + i]
                        - workspace->s[system->N + j];
            e_ele = -0.5 * KCALpMOL_to_EV * bond_softness
                         * SQR( effpot_diff );

            data->my_en.e_ele += e_ele;

            /* forces contribution */
            d_bond_softness = system->reax_param.gp.l[34]
                            * 3.0 / xcut * pow( d, 2.0 )
                            * pow( 1.0 - d, 5.0 ) * (1.0 - 3.0 * d);
            d_bond_softness = -0.5 * d_bond_softness
                            * SQR( effpot_diff );
            d_bond_softness = KCALpMOL_to_EV * d_bond_softness
                            / nbr_pj->d;

            /* tally into per-atom energy */
            if (system->pair_ptr->evflag || system->pair_ptr->vflag_atom) {
              rvec_ScaledSum( delij, 1., system->my_atoms[i].x,
                                    -1., system->my_atoms[j].x );
              f_tmp = -d_bond_softness;
              system->pair_ptr->ev_tally(i,j,natoms,1,0.0,e_ele,
                                f_tmp,delij[0],delij[1],delij[2]);
            }

            rvec_ScaledAdd( workspace->f[i], -d_bond_softness, nbr_pj->dvec );
            rvec_ScaledAdd( workspace->f[j], d_bond_softness, nbr_pj->dvec );
          }
        }
      }
    }

    Compute_Polarization_Energy(system, data, workspace);
  }

  void LR_vdW_Coulomb(reax_system *system, storage *workspace,
                      control_params *control, int i, int j,
                      double r_ij, LR_data *lr)
  {
    double p_vdW1 = system->reax_param.gp.l[28];
    double p_vdW1i = 1.0 / p_vdW1;
    double powr_vdW1, powgi_vdW1;
    double tmp, fn13, exp1, exp2;
    double Tap, dTap, dfn13;
    double dr3gamij_1, dr3gamij_3;
    double e_core, de_core;
    double e_lg, de_lg, r_ij5, r_ij6, re6;
    two_body_parameters *twbp;

    twbp = &(system->reax_param.tbp[i][j]);
    e_core = 0;
    de_core = 0;
    e_lg = de_lg = 0.0;

    /* calculate taper and its derivative */
    Tap = workspace->Tap[7] * r_ij + workspace->Tap[6];
    Tap = Tap * r_ij + workspace->Tap[5];
    Tap = Tap * r_ij + workspace->Tap[4];
    Tap = Tap * r_ij + workspace->Tap[3];
    Tap = Tap * r_ij + workspace->Tap[2];
    Tap = Tap * r_ij + workspace->Tap[1];
    Tap = Tap * r_ij + workspace->Tap[0];

    dTap = 7*workspace->Tap[7] * r_ij + 6*workspace->Tap[6];
    dTap = dTap * r_ij + 5*workspace->Tap[5];
    dTap = dTap * r_ij + 4*workspace->Tap[4];
    dTap = dTap * r_ij + 3*workspace->Tap[3];
    dTap = dTap * r_ij + 2*workspace->Tap[2];
    dTap += workspace->Tap[1]/r_ij;

    /*vdWaals Calculations*/
    if (system->reax_param.gp.vdw_type==1 || system->reax_param.gp.vdw_type==3)
      { // shielding
        powr_vdW1 = pow(r_ij, p_vdW1);
        powgi_vdW1 = pow(1.0 / twbp->gamma_w, p_vdW1);

        fn13 = pow(powr_vdW1 + powgi_vdW1, p_vdW1i);
        exp1 = exp(twbp->alpha * (1.0 - fn13 / twbp->r_vdW));
        exp2 = exp(0.5 * twbp->alpha * (1.0 - fn13 / twbp->r_vdW));

        lr->e_vdW = Tap * twbp->D * (exp1 - 2.0 * exp2);

        dfn13 = pow(powr_vdW1 + powgi_vdW1, p_vdW1i-1.0) * pow(r_ij, p_vdW1-2.0);

        lr->CEvd = dTap * twbp->D * (exp1 - 2.0 * exp2) -
          Tap * twbp->D * (twbp->alpha / twbp->r_vdW) * (exp1 - exp2) * dfn13;
      }
    else { // no shielding
      exp1 = exp(twbp->alpha * (1.0 - r_ij / twbp->r_vdW));
      exp2 = exp(0.5 * twbp->alpha * (1.0 - r_ij / twbp->r_vdW));

      lr->e_vdW = Tap * twbp->D * (exp1 - 2.0 * exp2);
      lr->CEvd = dTap * twbp->D * (exp1 - 2.0 * exp2) -
        Tap * twbp->D * (twbp->alpha / twbp->r_vdW) * (exp1 - exp2) / r_ij;
    }

    if (system->reax_param.gp.vdw_type==2 || system->reax_param.gp.vdw_type==3)
      { // inner wall
        e_core = twbp->ecore * exp(twbp->acore * (1.0-(r_ij/twbp->rcore)));
        lr->e_vdW += Tap * e_core;

        de_core = -(twbp->acore/twbp->rcore) * e_core;
        lr->CEvd += dTap * e_core + Tap * de_core / r_ij;

        //  lg correction, only if lgvdw is yes
        if (control->lgflag) {
          r_ij5 = pow(r_ij, 5.0);
          r_ij6 = pow(r_ij, 6.0);
          re6 = pow(twbp->lgre, 6.0);
          e_lg = -(twbp->lgcij/(r_ij6 + re6));
          lr->e_vdW += Tap * e_lg;

          de_lg = -6.0 * e_lg *  r_ij5 / (r_ij6 + re6) ;
          lr->CEvd += dTap * e_lg + Tap * de_lg/r_ij;
        }

      }


    /* Coulomb calculations */
    dr3gamij_1 = (r_ij * r_ij * r_ij + twbp->gamma);
    dr3gamij_3 = pow(dr3gamij_1 , 0.33333333333333);

    tmp = Tap / dr3gamij_3;
    lr->H = EV_to_KCALpMOL * tmp;
    lr->e_ele = C_ele * tmp;

    lr->CEclmb = C_ele * (dTap -  Tap * r_ij / dr3gamij_1) / dr3gamij_3;
  }
}

