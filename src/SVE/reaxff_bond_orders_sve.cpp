// clang-format off
/*----------------------------------------------------------------------
  PuReMD - Purdue ReaxFFSVe Molecular Dynamics Program

  Copyright (2010) Purdue University

  Contributing authors:
  H. M. Aktulga, J. Fogarty, S. Pandit, A. Grama
  Corresponding author:
  Hasan Metin Aktulga, Michigan State University, hma@cse.msu.edu

  Please cite the related publication:
  H. M. Aktulga, J. C. Fogarty, S. A. Pandit, A. Y. Grama,
  "Parallel Reactive Molecular Dynamics: Numerical Methods and
  Algorithmic Techniques", Parallel Computing, 38 (4-5), 245-259

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
#include "arm_sve.h"
#include "sve_util.h"

#include <cmath>
#include <omp.h>

#define BLKSIZE 8

namespace ReaxFFSVE {
  void Add_dBond_to_Forces(reax_system *system, int i, int pj, storage *workspace, reax_list **lists)
  {
    reax_list *bonds = (*lists) + BONDS;
    bond_data *nbr_j, *nbr_k;
    bond_order_data *bo_ij, *bo_ji;
    dbond_coefficients coef;
    int pk, k, j;

    /* Virial Tallying variables */
    rvec fi_tmp, fj_tmp, fk_tmp, delij, delji, delki, delkj, temp;

    /* Initializations */
    nbr_j = &(bonds->select.bond_list[pj]);
    j = nbr_j->nbr;
    bo_ij = &(nbr_j->bo_data);
    bo_ji = &(bonds->select.bond_list[nbr_j->sym_index].bo_data);

    double c = bo_ij->Cdbo + bo_ji->Cdbo;
    coef.C1dbo = bo_ij->C1dbo * c;
    coef.C2dbo = bo_ij->C2dbo * c;
    coef.C3dbo = bo_ij->C3dbo * c;

    c = bo_ij->Cdbopi + bo_ji->Cdbopi;
    coef.C1dbopi = bo_ij->C1dbopi * c;
    coef.C2dbopi = bo_ij->C2dbopi * c;
    coef.C3dbopi = bo_ij->C3dbopi * c;
    coef.C4dbopi = bo_ij->C4dbopi * c;

    c = bo_ij->Cdbopi2 + bo_ji->Cdbopi2;
    coef.C1dbopi2 = bo_ij->C1dbopi2 * c;
    coef.C2dbopi2 = bo_ij->C2dbopi2 * c;
    coef.C3dbopi2 = bo_ij->C3dbopi2 * c;
    coef.C4dbopi2 = bo_ij->C4dbopi2 * c;

    c = workspace->CdDelta[i] + workspace->CdDelta[j];
    coef.C1dDelta = bo_ij->C1dbo * c;
    coef.C2dDelta = bo_ij->C2dbo * c;
    coef.C3dDelta = bo_ij->C3dbo * c;

    c = (coef.C1dbo + coef.C1dDelta + coef.C2dbopi + coef.C2dbopi2);
    rvec_Scale(    temp, c,    bo_ij->dBOp);

    c = (coef.C2dbo + coef.C2dDelta + coef.C3dbopi + coef.C3dbopi2);
    rvec_ScaledAdd(temp, c,    workspace->dDeltap_self[i]);

    rvec_ScaledAdd(temp, coef.C1dbopi,  bo_ij->dln_BOp_pi);
    rvec_ScaledAdd(temp, coef.C1dbopi2, bo_ij->dln_BOp_pi2);

    rvec_Add(workspace->f[i], temp);

    if (system->pair_ptr->vflag_either) {
      rvec_Scale(fi_tmp, -0.5, temp);
      rvec_ScaledSum(delij, 1., system->my_atoms[i].x,-1., system->my_atoms[j].x);
      system->pair_ptr->v_tally2_newton(i,fi_tmp,delij);
    }

    c = -(coef.C1dbo + coef.C1dDelta + coef.C2dbopi + coef.C2dbopi2);
    rvec_Scale(    temp, c,    bo_ij->dBOp);

    c = (coef.C3dbo + coef.C3dDelta + coef.C4dbopi + coef.C4dbopi2);
    rvec_ScaledAdd(temp,  c,    workspace->dDeltap_self[j]);

    rvec_ScaledAdd(temp, -coef.C1dbopi,  bo_ij->dln_BOp_pi);
    rvec_ScaledAdd(temp, -coef.C1dbopi2, bo_ij->dln_BOp_pi2);

    rvec_Add(workspace->f[j], temp);

    if (system->pair_ptr->vflag_either) {
      rvec_Scale(fj_tmp, -0.5, temp);
      rvec_ScaledSum(delji, 1., system->my_atoms[j].x,-1., system->my_atoms[i].x);
      system->pair_ptr->v_tally2_newton(j,fj_tmp,delji);
    }

    // forces on k: i neighbor
    for (pk = Start_Index(i, bonds); pk < End_Index(i, bonds); ++pk) {
      nbr_k = &(bonds->select.bond_list[pk]);
      k = nbr_k->nbr;

      const double c = -(coef.C2dbo + coef.C2dDelta + coef.C3dbopi + coef.C3dbopi2);
      rvec_Scale(temp, c, nbr_k->bo_data.dBOp);

      rvec_Add(workspace->f[k], temp);

      if (system->pair_ptr->vflag_either) {
        rvec_Scale(fk_tmp, -0.5, temp);
        rvec_ScaledSum(delki,1.,system->my_atoms[k].x,-1.,system->my_atoms[i].x);
        system->pair_ptr->v_tally2_newton(k,fk_tmp,delki);
        rvec_ScaledSum(delkj,1.,system->my_atoms[k].x,-1.,system->my_atoms[j].x);
        system->pair_ptr->v_tally2_newton(k,fk_tmp,delkj);
      }
    }

    // forces on k: j neighbor
    for (pk = Start_Index(j, bonds); pk < End_Index(j, bonds); ++pk) {
      nbr_k = &(bonds->select.bond_list[pk]);
      k = nbr_k->nbr;

      const double c = -(coef.C3dbo + coef.C3dDelta + coef.C4dbopi + coef.C4dbopi2);
      rvec_Scale(temp, c, nbr_k->bo_data.dBOp);

      rvec_Add(workspace->f[k], temp);

      if (system->pair_ptr->vflag_either) {
        rvec_Scale(fk_tmp, -0.5, temp);
        rvec_ScaledSum(delki,1.,system->my_atoms[k].x,-1.,system->my_atoms[i].x);
        system->pair_ptr->v_tally2_newton(k,fk_tmp,delki);
        rvec_ScaledSum(delkj,1.,system->my_atoms[k].x,-1.,system->my_atoms[j].x);
        system->pair_ptr->v_tally2_newton(k,fk_tmp,delkj);
      }
    }
  }


  /* modify by szh */

  typedef struct {
    int cnt;
    double Deltap_boc_i[BLKSIZE];
    double Deltap_boc_j[BLKSIZE];
    double BO[BLKSIZE];
    double BO_pi[BLKSIZE];
    double BO_pi2[BLKSIZE];
    double p_boc3[BLKSIZE];
    double p_boc4[BLKSIZE];
    double p_boc5[BLKSIZE];
    int64_t i[BLKSIZE];
    long addr[BLKSIZE];
  } flag1_buffer_t;
  
  typedef struct {
    int cnt;
    double val_i[BLKSIZE];
    double val_j[BLKSIZE];
    double Deltap_i[BLKSIZE];
    double Deltap_j[BLKSIZE];
    double Deltap_boc_i[BLKSIZE];
    double Deltap_boc_j[BLKSIZE];
    double BO[BLKSIZE];
    double BO_pi[BLKSIZE];
    double BO_pi2[BLKSIZE];
    double p_boc3[BLKSIZE];
    double p_boc4[BLKSIZE];
    double p_boc5[BLKSIZE];
    int64_t i[BLKSIZE];
    long addr[BLKSIZE];
  } flag2_buffer_t;

  typedef struct {
    int cnt;
    double val_i[BLKSIZE];
    double val_j[BLKSIZE];
    double Deltap_i[BLKSIZE];
    double Deltap_j[BLKSIZE];
    double BO[BLKSIZE];
    double BO_pi[BLKSIZE];
    double BO_pi2[BLKSIZE];
    int64_t i[BLKSIZE];
    long addr[BLKSIZE];
  } flag3_buffer_t;

  void buffer_bo_flag1(flag1_buffer_t *fb, reax_system *system, storage *workspace)
  {
    svd_t vone = svdup_f64(1.0);
    svd_t vtwo = svdup_f64(2.0);

    svbool_t predi = svwhilelt_b64(0, fb->cnt); 
    svd_t vDeltap_boc_i = svld1_f64(predi, fb->Deltap_boc_i);
    svd_t vDeltap_boc_j = svld1_f64(predi, fb->Deltap_boc_j);
    svd_t vBO = svld1_f64(predi, fb->BO);
    svd_t vBO_pi = svld1_f64(predi, fb->BO_pi);
    svd_t vBO_pi2 = svld1_f64(predi, fb->BO_pi2);
    svd_t vp_boc3 = svld1_f64(predi, fb->p_boc3);
    svd_t vp_boc4 = svld1_f64(predi, fb->p_boc4);
    svd_t vp_boc5 = svld1_f64(predi, fb->p_boc5);
    svsl_t vi = svld1_s64(predi, fb->i);
    svsl_t vaddr = svld1_s64(predi, fb->addr);

    /* Correction for 1-3 bond orders */
    svd_t vp_boc4_times_vBO = vp_boc4 * vBO * vBO;
    svd_t vexp_fs[2] = {-(vp_boc4_times_vBO - vDeltap_boc_i) * vp_boc3 + vp_boc5,
                        -(vp_boc4_times_vBO - vDeltap_boc_j) * vp_boc3 + vp_boc5};
    svd_t vexp_fs_result[2];
    svnxp_exp<2, 8, true>(vexp_fs_result, vexp_fs);
    svd_t vexp_f4 = vexp_fs_result[0];
    svd_t vexp_f5 = vexp_fs_result[1];
    // test 0302
    // double vec_f4[8], vec_f5[8];
    // svst1(predi, vec_f4, vexp_f4);
    // svst1(predi, vec_f5, vexp_f5);
    // for (int xx = 0; xx < 8; xx++) {
    //   double scalar_f4 = exp(-(fb->p_boc4[xx] * fb->BO[xx] * fb->BO[xx] - fb->Deltap_boc_i[xx]) * 
    //                       fb->p_boc3[xx] + fb->p_boc5[xx]);
    //   double scalar_f5 = exp(-(fb->p_boc4[xx] * fb->BO[xx] * fb->BO[xx] - fb->Deltap_boc_j[xx]) * 
    //                       fb->p_boc3[xx] + fb->p_boc5[xx]);
    //   printf("  f4 -> Vector: %.10f | Scalar: %.10f | Diff: %e\n", 
    //         vec_f4[xx], scalar_f4, vec_f4[xx] - scalar_f4);
    //   printf("  f5 -> Vector: %.10f | Scalar: %.10f | Diff: %e\n", 
    //           vec_f5[xx], scalar_f5, vec_f5[xx] - scalar_f5);
    // }

    svd_t vf4 = vone / (vone + vexp_f4);
    svd_t vf5 = vone / (vone + vexp_f5);
    svd_t vf4f5 = vf4 * vf5;

    /* Bond Order pages 8-9, derivative of f4 and f5 */
    svd_t vCf45_ij = -vf4 * vexp_f4;
    svd_t vCf45_ji = -vf5 * vexp_f5;

    /* Bond Order page 10, derivative of total bond order */
    svd_t vA1_ij = -vtwo * vp_boc3 * vp_boc4 * vBO * (vCf45_ij + vCf45_ji);
    svd_t vA2_ij = vp_boc3 * vCf45_ij;
    svd_t vA2_ji = vp_boc3 * vCf45_ji;
    svd_t vA3_ij = vA2_ij;
    svd_t vA3_ji = vA2_ji;

    /* find corrected bond orders and their derivative coef */
    vBO     = vBO * vf4f5;
    vBO_pi  = vBO_pi * vf4f5;
    vBO_pi2 = vBO_pi2 * vf4f5;
    svd_t vBO_s = vBO - vBO_pi - vBO_pi2;
    svd_t vC1dbo = vf4f5 + vBO * vA1_ij;
    svd_t vC2dbo = vBO * vA2_ij;
    svd_t vC3dbo = vBO * vA2_ji;
    svd_t vC2dbopi = vBO_pi * vA1_ij;
    svd_t vC3dbopi = vBO_pi * vA3_ij;
    svd_t vC4dbopi = vBO_pi * vA3_ji;
    svd_t vC2dbopi2 = vBO_pi2 * vA1_ij;
    svd_t vC3dbopi2 = vBO_pi2 * vA3_ij;
    svd_t vC4dbopi2 = vBO_pi2 * vA3_ji;
    
    svst1_scatter_offset(predi, 0, vaddr, vBO);
    svst1_scatter_offset(predi, 0, vaddr + 8, vBO_s);
    svst1_scatter_offset(predi, 0, vaddr + 16, vBO_pi);
    svst1_scatter_offset(predi, 0, vaddr + 24, vBO_pi2);
    svst1_scatter_offset(predi, 0, vaddr + 56, vC1dbo);
    svst1_scatter_offset(predi, 0, vaddr + 64, vC2dbo);
    svst1_scatter_offset(predi, 0, vaddr + 72, vC3dbo);
    svst1_scatter_offset(predi, 0, vaddr + 80, vf4f5);
    svst1_scatter_offset(predi, 0, vaddr + 88, vC2dbopi);
    svst1_scatter_offset(predi, 0, vaddr + 96, vC3dbopi);
    svst1_scatter_offset(predi, 0, vaddr + 104, vC4dbopi);
    svst1_scatter_offset(predi, 0, vaddr + 112, vf4f5);
    svst1_scatter_offset(predi, 0, vaddr + 120, vC2dbopi2);
    svst1_scatter_offset(predi, 0, vaddr + 128, vC3dbopi2);
    svst1_scatter_offset(predi, 0, vaddr + 136, vC4dbopi2);
    while (svptest_any(predi, predi)) {
      int64_t rep_i = svlastb(predi, vi);
      svbool_t mask = svcmpeq(predi, vi, rep_i);
      workspace->total_bond_order[rep_i] += svaddv_f64(mask, vBO);
      predi = svbic_z(predi, predi, mask);
    }
  }

  void buffer_bo_flag2(flag2_buffer_t *fb, reax_system *system, storage *workspace, svd_t vp_boc1, svd_t vp_boc2)
  {
    svd_t vhalf = svdup_f64(0.5);
    svd_t vone = svdup_f64(1.0);
    svd_t vtwo = svdup_f64(2.0);

    svbool_t predi = svwhilelt_b64(0, fb->cnt); 
    svd_t vval_i = svld1_f64(predi, fb->val_i);
    svd_t vval_j = svld1_f64(predi, fb->val_j);
    svd_t vDeltap_i = svld1_f64(predi, fb->Deltap_i);
    svd_t vDeltap_j = svld1_f64(predi, fb->Deltap_j);
    svd_t vDeltap_boc_i = svld1_f64(predi, fb->Deltap_boc_i);
    svd_t vDeltap_boc_j = svld1_f64(predi, fb->Deltap_boc_j);
    svd_t vBO = svld1_f64(predi, fb->BO);
    svd_t vBO_pi = svld1_f64(predi, fb->BO_pi);
    svd_t vBO_pi2 = svld1_f64(predi, fb->BO_pi2);
    svd_t vp_boc3 = svld1_f64(predi, fb->p_boc3);
    svd_t vp_boc4 = svld1_f64(predi, fb->p_boc4);
    svd_t vp_boc5 = svld1_f64(predi, fb->p_boc5);
    svsl_t vi = svld1_s64(predi, fb->i);
    svsl_t vaddr = svld1_s64(predi, fb->addr);

    /* Correction for overcoordination */
    svd_t vexp_ps[4] = {-vp_boc1 * vDeltap_i, -vp_boc1 * vDeltap_j,
                        -vp_boc2 * vDeltap_i, -vp_boc2 * vDeltap_j};
    svd_t vexp_ps_result[4];
    svnxp_exp<4, 8, true>(vexp_ps_result, vexp_ps);
    svd_t vexp_p1i = vexp_ps_result[0];
    svd_t vexp_p1j = vexp_ps_result[1];

    svd_t vexp_p2i = vexp_ps_result[2];
    svd_t vexp_p2j = vexp_ps_result[3];
    svd_t vexp_p2_sum = vexp_p2i + vexp_p2j;
    svd_t vexp_p2i_p2j_inv = vone / vexp_p2_sum;

    svd_t vf3_log[1] = {vhalf * vexp_p2_sum};
    svd_t vf3_log_result[1];
    svnxp_log<1, 8>(vf3_log_result, vf3_log);
    svd_t vf2 = vexp_p1i + vexp_p1j;
    svd_t vf3 = -vf3_log_result[0] / vp_boc2;

    svd_t vval_i_plus_vf2 = vval_i + vf2;
    svd_t vval_j_plus_vf2 = vval_j + vf2;
    svd_t vval_i_plus_vf2_plus_vf3 = vval_i_plus_vf2 + vf3;
    svd_t vval_j_plus_vf2_plus_vf3 = vval_j_plus_vf2 + vf3;
    svd_t vf1 = vhalf * (vval_i_plus_vf2 / vval_i_plus_vf2_plus_vf3 + vval_j_plus_vf2 / vval_j_plus_vf2_plus_vf3);

    svd_t vtemp = vf2 + vf3;
    svd_t vu1_ij = vval_i + vtemp;
    svd_t vu1_ji = vval_j + vtemp;
    svd_t vu1_ij_inv = vone / vu1_ij;
    svd_t vu1_ji_inv = vone / vu1_ji;

    svd_t vu1_ij_inv_sq = vu1_ij_inv * vu1_ij_inv;
    svd_t vu1_ji_inv_sq = vu1_ji_inv * vu1_ji_inv;

    svd_t vCf1A_ij = vhalf * vf3 * (vu1_ij_inv_sq + vu1_ji_inv_sq);
    svd_t vCf1B_ij = -vhalf * ((vu1_ij - vf3) * vu1_ij_inv_sq  + (vu1_ji - vf3) * vu1_ji_inv_sq);

    svd_t vexp_p1i_times_vp_boc1 = vp_boc1 * vexp_p1i;
    svd_t vexp_p1j_times_vp_boc1 = vp_boc1 * vexp_p1j;
    svd_t vexp_p2i_times_inv = vexp_p2i * vexp_p2i_p2j_inv;
    svd_t vexp_p2j_times_inv = vexp_p2j * vexp_p2i_p2j_inv;

    svd_t vterm_i = -vexp_p1i_times_vp_boc1 + vexp_p2i_times_inv;
    svd_t vCf1_ij = vhalf * (
        -vexp_p1i_times_vp_boc1 * (vu1_ij_inv + vu1_ji_inv) -
        vval_i_plus_vf2 * vu1_ij_inv_sq * vterm_i -
        vval_j_plus_vf2 * vu1_ji_inv_sq * vterm_i
    );

    svd_t vCf1_ji = -vCf1A_ij * vexp_p1j_times_vp_boc1 + vCf1B_ij * vexp_p2j_times_inv;

    svd_t vf1_inv = vone / vf1;

    /* Correction for 1-3 bond orders */
    svd_t vp_boc4_times_vBO = vp_boc4 * vBO * vBO;
    svd_t vexp_fs[2] = {- (vp_boc4_times_vBO - vDeltap_boc_i) * vp_boc3 + vp_boc5,
                        - (vp_boc4_times_vBO - vDeltap_boc_j) * vp_boc3 + vp_boc5};
    svd_t vexp_fs_result[2];
    svnxp_exp<2, 8>(vexp_fs_result, vexp_fs);
    svd_t vexp_f4 = vexp_fs_result[0];
    svd_t vexp_f5 = vexp_fs_result[1];

    svd_t vf4 = vone / (vone + vexp_f4);
    svd_t vf5 = vone / (vone + vexp_f5);
    svd_t vf4f5 = vf4 * vf5;

    /* Bond Order pages 8-9, derivative of f4 and f5 */
    svd_t vCf45_ij = -vf4 * vexp_f4;
    svd_t vCf45_ji = -vf5 * vexp_f5;

    /* Bond Order page 10, derivative of total bond order */
    svd_t vCf1_ij_times_f1_inv = vCf1_ij * vf1_inv;
    svd_t vCf1_ji_times_f1_inv = vCf1_ji * vf1_inv;
    svd_t vA0_ij = vf1 * vf4f5;
    svd_t vA1_ij = -vtwo * vp_boc3 * vp_boc4 * vBO * (vCf45_ij + vCf45_ji);
    svd_t vA2_ij = vCf1_ij_times_f1_inv + vp_boc3 * vCf45_ij;
    svd_t vA2_ji = vCf1_ji_times_f1_inv + vp_boc3 * vCf45_ji;
    svd_t vA3_ij = vA2_ij + vCf1_ij_times_f1_inv;
    svd_t vA3_ji = vA2_ji + vCf1_ji_times_f1_inv;

    /* find corrected bond orders and their derivative coef */
    vBO     = vBO * vA0_ij;
    vBO_pi  = vBO_pi * vA0_ij * vf1;
    vBO_pi2 = vBO_pi2 * vA0_ij * vf1;
    svd_t vBO_s = vBO - vBO_pi - vBO_pi2;
    svd_t vC1dbo = vA0_ij + vBO * vA1_ij;
    svd_t vC2dbo = vBO * vA2_ij;
    svd_t vC3dbo = vBO * vA2_ji;
    svd_t vC1dbopi = vf1 * vf1 * vf4f5;
    svd_t vC2dbopi = vBO_pi * vA1_ij;
    svd_t vC3dbopi = vBO_pi * vA3_ij;
    svd_t vC4dbopi = vBO_pi * vA3_ji;
    svd_t vC1dbopi2 = vC1dbopi;
    svd_t vC2dbopi2 = vBO_pi2 * vA1_ij;
    svd_t vC3dbopi2 = vBO_pi2 * vA3_ij;
    svd_t vC4dbopi2 = vBO_pi2 * vA3_ji;
    
    svst1_scatter_offset(predi, 0, vaddr, vBO);
    svst1_scatter_offset(predi, 0, vaddr + 8, vBO_s);
    svst1_scatter_offset(predi, 0, vaddr + 16, vBO_pi);
    svst1_scatter_offset(predi, 0, vaddr + 24, vBO_pi2);
    svst1_scatter_offset(predi, 0, vaddr + 56, vC1dbo);
    svst1_scatter_offset(predi, 0, vaddr + 64, vC2dbo);
    svst1_scatter_offset(predi, 0, vaddr + 72, vC3dbo);
    svst1_scatter_offset(predi, 0, vaddr + 80, vC1dbopi);
    svst1_scatter_offset(predi, 0, vaddr + 88, vC2dbopi);
    svst1_scatter_offset(predi, 0, vaddr + 96, vC3dbopi);
    svst1_scatter_offset(predi, 0, vaddr + 104, vC4dbopi);
    svst1_scatter_offset(predi, 0, vaddr + 112, vC1dbopi2);
    svst1_scatter_offset(predi, 0, vaddr + 120, vC2dbopi2);
    svst1_scatter_offset(predi, 0, vaddr + 128, vC3dbopi2);
    svst1_scatter_offset(predi, 0, vaddr + 136, vC4dbopi2);
    while (svptest_any(predi, predi)) {
      int64_t rep_i = svlastb(predi, vi);
      svbool_t mask = svcmpeq(predi, vi, rep_i);
      workspace->total_bond_order[rep_i] += svaddv_f64(mask, vBO);
      predi = svbic_z(predi, predi, mask);
    }
  }

  void buffer_bo_flag3(flag3_buffer_t *fb, reax_system *system, storage *workspace, svd_t vp_boc1, svd_t vp_boc2)
  {
    svd_t vzero = svdup_f64(0.0);
    svd_t vhalf = svdup_f64(0.5);
    svd_t vone = svdup_f64(1.0);
    svd_t vtwo = svdup_f64(2.0);

    svbool_t predi = svwhilelt_b64(0, fb->cnt); 
    svd_t vval_i = svld1_f64(predi, fb->val_i);
    svd_t vval_j = svld1_f64(predi, fb->val_j);
    svd_t vDeltap_i = svld1_f64(predi, fb->Deltap_i);
    svd_t vDeltap_j = svld1_f64(predi, fb->Deltap_j);
    svd_t vBO = svld1_f64(predi, fb->BO);
    svd_t vBO_pi = svld1_f64(predi, fb->BO_pi);
    svd_t vBO_pi2 = svld1_f64(predi, fb->BO_pi2);
    svsl_t vi = svld1_s64(predi, fb->i);
    svsl_t vaddr = svld1_s64(predi, fb->addr);

    /* Correction for overcoordination */
    svd_t vexp_ps[4] = {-vp_boc1 * vDeltap_i, -vp_boc1 * vDeltap_j,
                        -vp_boc2 * vDeltap_i, -vp_boc2 * vDeltap_j};
    svd_t vexp_ps_result[4];
    svnxp_exp<4, 8, true>(vexp_ps_result, vexp_ps);
    svd_t vexp_p1i = vexp_ps_result[0];
    svd_t vexp_p1j = vexp_ps_result[1];

    svd_t vexp_p2i = vexp_ps_result[2];
    svd_t vexp_p2j = vexp_ps_result[3];
    svd_t vexp_p2_sum = vexp_p2i + vexp_p2j;
    svd_t vexp_p2i_p2j_inv = vone / vexp_p2_sum;

    svd_t vf3_log[1] = {vhalf * vexp_p2_sum};
    svd_t vf3_log_result[1];
    svnxp_log<1, 8>(vf3_log_result, vf3_log);
    svd_t vf2 = vexp_p1i + vexp_p1j;
    svd_t vf3 = -vf3_log_result[0] / vp_boc2;

    svd_t vval_i_plus_vf2 = vval_i + vf2;
    svd_t vval_j_plus_vf2 = vval_j + vf2;
    svd_t vval_i_plus_vf2_plus_vf3 = vval_i_plus_vf2 + vf3;
    svd_t vval_j_plus_vf2_plus_vf3 = vval_j_plus_vf2 + vf3;
    svd_t vf1 = vhalf * (vval_i_plus_vf2 / vval_i_plus_vf2_plus_vf3 + vval_j_plus_vf2 / vval_j_plus_vf2_plus_vf3);

    svd_t vtemp = vf2 + vf3;
    svd_t vu1_ij = vval_i + vtemp;
    svd_t vu1_ji = vval_j + vtemp;
    svd_t vu1_ij_inv = vone / vu1_ij;
    svd_t vu1_ji_inv = vone / vu1_ji;

    svd_t vu1_ij_inv_sq = vu1_ij_inv * vu1_ij_inv;
    svd_t vu1_ji_inv_sq = vu1_ji_inv * vu1_ji_inv;

    svd_t vCf1A_ij = vhalf * vf3 * (vu1_ij_inv_sq + vu1_ji_inv_sq);
    svd_t vCf1B_ij = -vhalf * ((vu1_ij - vf3) * vu1_ij_inv_sq  + (vu1_ji - vf3) * vu1_ji_inv_sq);

    svd_t vexp_p1i_times_vp_boc1 = vp_boc1 * vexp_p1i;
    svd_t vexp_p1j_times_vp_boc1 = vp_boc1 * vexp_p1j;
    svd_t vexp_p2i_times_inv = vexp_p2i * vexp_p2i_p2j_inv;
    svd_t vexp_p2j_times_inv = vexp_p2j * vexp_p2i_p2j_inv;

    svd_t vterm_i = -vexp_p1i_times_vp_boc1 + vexp_p2i_times_inv;
    svd_t vCf1_ij = vhalf * (
        -vexp_p1i_times_vp_boc1 * (vu1_ij_inv + vu1_ji_inv) -
        vval_i_plus_vf2 * vu1_ij_inv_sq * vterm_i -
        vval_j_plus_vf2 * vu1_ji_inv_sq * vterm_i
    );

    svd_t vCf1_ji = -vCf1A_ij * vexp_p1j_times_vp_boc1 + vCf1B_ij * vexp_p2j_times_inv;

    svd_t vf1_inv = vone / vf1;

    svd_t vA2_ij = vCf1_ij;
    svd_t vA2_ji = vCf1_ji;
    svd_t vA3_ij = vA2_ij + vCf1_ij * vf1_inv;
    svd_t vA3_ji = vA2_ji + vCf1_ij * vf1_inv;

    /* find corrected bond orders and their derivative coef */
    vBO     = vBO * vf1;
    vBO_pi  = vBO_pi * vf1 * vf1;
    vBO_pi2 = vBO_pi2 * vf1 * vf1;
    svd_t vBO_s = vBO - vBO_pi - vBO_pi2;
    svd_t vC2dbo = vBO * vA2_ij;
    svd_t vC3dbo = vBO * vA2_ji;
    svd_t vC3dbopi = vBO_pi * vA3_ij;
    svd_t vC4dbopi = vBO_pi * vA3_ji;
    svd_t vC3dbopi2 = vBO_pi2 * vA3_ij;
    svd_t vC4dbopi2 = vBO_pi2 * vA3_ji;
    
    svst1_scatter_offset(predi, 0, vaddr, vBO);
    svst1_scatter_offset(predi, 0, vaddr + 8, vBO_s);
    svst1_scatter_offset(predi, 0, vaddr + 16, vBO_pi);
    svst1_scatter_offset(predi, 0, vaddr + 24, vBO_pi2);
    svst1_scatter_offset(predi, 0, vaddr + 56, vf1);
    svst1_scatter_offset(predi, 0, vaddr + 64, vC2dbo);
    svst1_scatter_offset(predi, 0, vaddr + 72, vC3dbo);
    svst1_scatter_offset(predi, 0, vaddr + 80, vzero);
    svst1_scatter_offset(predi, 0, vaddr + 88, vzero);
    svst1_scatter_offset(predi, 0, vaddr + 96, vC3dbopi);
    svst1_scatter_offset(predi, 0, vaddr + 104, vC4dbopi);
    svst1_scatter_offset(predi, 0, vaddr + 112, vzero);
    svst1_scatter_offset(predi, 0, vaddr + 120, vzero);
    svst1_scatter_offset(predi, 0, vaddr + 128, vC3dbopi2);
    svst1_scatter_offset(predi, 0, vaddr + 136, vC4dbopi2);
    while (svptest_any(predi, predi)) {
      int64_t rep_i = svlastb(predi, vi);
      svbool_t mask = svcmpeq(predi, vi, rep_i);
      workspace->total_bond_order[rep_i] += svaddv_f64(mask, vBO);
      predi = svbic_z(predi, predi, mask);
    }
  }


  void BO(reax_system *system, storage *workspace, reax_list **lists)
  {
    int i, j, pj, type_i, type_j;
    int start_i, end_i, sym_index;
    double val_i, Deltap_i, Deltap_boc_i;
    double val_j, Deltap_j, Deltap_boc_j;
    // double f1, f2, f3, f4, f5, f4f5, exp_f4, exp_f5;
    // double exp_p1i, exp_p2i, exp_p1j, exp_p2j;
    // double temp, u1_ij, u1_ji, Cf1A_ij, Cf1B_ij, Cf1_ij, Cf1_ji;
    // double Cf45_ij, Cf45_ji, p_lp1; //u_ij, u_ji
    // double A0_ij, A1_ij, A2_ij, A2_ji, A3_ij, A3_ji;
    // double explp1;
    double p_lp1, explp1;
    single_body_parameters *sbp_i, *sbp_j;
    two_body_parameters *twbp;
    bond_order_data *bo_ij, *bo_ji;
    reax_list *bonds = (*lists) + BONDS;

    svd_t vp_boc1 = svdup_f64(system->reax_param.gp.l[0]);
    svd_t vp_boc2 = svdup_f64(system->reax_param.gp.l[1]);

    /* Calculate Deltaprime, Deltaprime_boc values */
  #pragma omp parallel for
    for (int i = 0; i < system->N; ++i) {
      int type_i = system->my_atoms[i].type;
      if (type_i < 0) continue;
      single_body_parameters *sbp_i = &(system->reax_param.sbp[type_i]);
      workspace->Deltap[i] = workspace->total_bond_order[i] - sbp_i->valency;
      workspace->Deltap_boc[i] = workspace->total_bond_order[i] - sbp_i->valency_val;

      workspace->total_bond_order[i] = 0;
    }


    // usage for sve
    flag1_buffer_t flag1;
    flag2_buffer_t flag2;
    flag3_buffer_t flag3;
    int nthr = omp_get_max_threads();

#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < system->N; ++i) {
      int start_i = bonds->index[i];
      int end_i = bonds->end_index[i];
      
      for (int pj = start_i; pj < end_i; ++pj) {
        int j = bonds->select.bond_list[pj].nbr;
        
        if (i < j) {
          int start_j = bonds->index[j];
          int end_j = bonds->end_index[j];
          
          for (int pk = start_j; pk < end_j; pk++) {
            int k = bonds->select.bond_list[pk].nbr;
            if (k == i) {
              bonds->select.bond_list[pj].sym_index = pk;
              bonds->select.bond_list[pk].sym_index = pj;
              break;
            }
          }
        }
      }
    }
    
    /* Corrected Bond Order calculations */
    typedef struct {
        flag1_buffer_t flag1;
        flag2_buffer_t flag2; 
        flag3_buffer_t flag3;
    } ThreadBuffers;

    std::vector<ThreadBuffers> thread_buffers(nthr);
    
  #pragma omp parallel
  {
    int tid = omp_get_thread_num();

    thread_buffers[tid].flag1.cnt = 0;
    thread_buffers[tid].flag2.cnt = 0;
    thread_buffers[tid].flag3.cnt = 0;

    ThreadBuffers *my_buffers = &thread_buffers[tid];
    
    #pragma omp for schedule(dynamic)
    for (int i = 0; i < system->N; ++i) {
      if (workspace->bond_mark[i] > 3) continue;
      int type_i = system->my_atoms[i].type;
      single_body_parameters *sbp_i = &(system->reax_param.sbp[type_i]);
      double val_i = sbp_i->valency;
      double Deltap_i = workspace->Deltap[i];
      double Deltap_boc_i = workspace->Deltap_boc[i];
      int start_i = bonds->index[i];
      int end_i = bonds->end_index[i];

      for (int pj = start_i; pj < end_i; ++pj) {
        int j = bonds->select.bond_list[pj].nbr;
        int type_j = system->my_atoms[j].type;
        bond_order_data *bo_ij = &(bonds->select.bond_list[pj].bo_data);

        two_body_parameters *twbp = &(system->reax_param.tbp[type_i][type_j]);
        if (twbp->v13cor >= 0.001 && twbp->ovc < 0.001) {
          flag1_buffer_t *flag1 = &my_buffers->flag1;
          flag1->Deltap_boc_i[flag1->cnt] = workspace->Deltap_boc[i];
          flag1->Deltap_boc_j[flag1->cnt] = workspace->Deltap_boc[j];
          flag1->BO[flag1->cnt] = bo_ij->BO;
          flag1->BO_pi[flag1->cnt] = bo_ij->BO_pi;
          flag1->BO_pi2[flag1->cnt] = bo_ij->BO_pi2;
          flag1->p_boc3[flag1->cnt] = twbp->p_boc3;
          flag1->p_boc4[flag1->cnt] = twbp->p_boc4;
          flag1->p_boc5[flag1->cnt] = twbp->p_boc5;
          flag1->i[flag1->cnt] = i;
          flag1->addr[flag1->cnt++] = (long)(&bo_ij->BO);
          if (flag1->cnt == BLKSIZE) {
            buffer_bo_flag1(flag1, system, workspace);
            flag1->cnt = 0;
          }
          // val_j = system->reax_param.sbp[type_j].valency;
          // Deltap_j = workspace->Deltap[j];
          // Deltap_boc_j = workspace->Deltap_boc[j];

          // /* No overcoordination correction! */
          // double f1 = 1.0;
          // double Cf1_ij = 0.0;
          // double Cf1_ji = 0.0;

          // double f1_inv = 1.0 / f1;

          // /* Correction for 1-3 bond orders */
          // double exp_f4 =exp(-(twbp->p_boc4 * SQR(bo_ij->BO) -
          //               Deltap_boc_i) * twbp->p_boc3 + twbp->p_boc5);
          // double exp_f5 =exp(-(twbp->p_boc4 * SQR(bo_ij->BO) -
          //               Deltap_boc_j) * twbp->p_boc3 + twbp->p_boc5);

          // double f4 = 1. / (1. + exp_f4);
          // double f5 = 1. / (1. + exp_f5);
          // double f4f5 = f4 * f5;

          // /* Bond Order pages 8-9, derivative of f4 and f5 */
          // double Cf45_ij = -f4 * exp_f4;
          // double Cf45_ji = -f5 * exp_f5;

          // /* Bond Order page 10, derivative of total bond order */
          // double A0_ij = f1 * f4f5;
          // double A1_ij = -2 * twbp->p_boc3 * twbp->p_boc4 * bo_ij->BO *
          //   (Cf45_ij + Cf45_ji);
          // double A2_ij = Cf1_ij *f1_inv + twbp->p_boc3 * Cf45_ij;
          // double A2_ji = Cf1_ji *f1_inv + twbp->p_boc3 * Cf45_ji;
          // double A3_ij = A2_ij + Cf1_ij *f1_inv;
          // double A3_ji = A2_ji + Cf1_ji *f1_inv;

          // /* find corrected bond orders and their derivative coef */
          // bo_ij->BO    = bo_ij->BO    * A0_ij;
          // bo_ij->BO_pi = bo_ij->BO_pi * A0_ij *f1;
          // bo_ij->BO_pi2= bo_ij->BO_pi2* A0_ij *f1;
          // bo_ij->BO_s  = bo_ij->BO - (bo_ij->BO_pi + bo_ij->BO_pi2);

          // bo_ij->C1dbo = A0_ij + bo_ij->BO * A1_ij;
          // bo_ij->C2dbo = bo_ij->BO * A2_ij;
          // bo_ij->C3dbo = bo_ij->BO * A2_ji;

          // bo_ij->C1dbopi = f1*f1*f4*f5;
          // bo_ij->C2dbopi = bo_ij->BO_pi * A1_ij;
          // bo_ij->C3dbopi = bo_ij->BO_pi * A3_ij;
          // bo_ij->C4dbopi = bo_ij->BO_pi * A3_ji;

          // bo_ij->C1dbopi2 = f1*f1*f4*f5;
          // bo_ij->C2dbopi2 = bo_ij->BO_pi2 * A1_ij;
          // bo_ij->C3dbopi2 = bo_ij->BO_pi2 * A3_ij;
          // bo_ij->C4dbopi2 = bo_ij->BO_pi2 * A3_ji;
        } else if (twbp->v13cor >= 0.001 && twbp->ovc >= 0.001) {
          flag2_buffer_t *flag2 = &my_buffers->flag2;
          flag2->val_i[flag2->cnt] = val_i;
          flag2->val_j[flag2->cnt] = system->reax_param.sbp[type_j].valency;
          flag2->Deltap_i[flag2->cnt] = workspace->Deltap[i];
          flag2->Deltap_j[flag2->cnt] = workspace->Deltap[j];
          flag2->Deltap_boc_i[flag2->cnt] = workspace->Deltap_boc[i];
          flag2->Deltap_boc_j[flag2->cnt] = workspace->Deltap_boc[j];
          flag2->BO[flag2->cnt] = bo_ij->BO;
          flag2->BO_pi[flag2->cnt] = bo_ij->BO_pi;
          flag2->BO_pi2[flag2->cnt] = bo_ij->BO_pi2;
          flag2->p_boc3[flag2->cnt] = twbp->p_boc3;
          flag2->p_boc4[flag2->cnt] = twbp->p_boc4;
          flag2->p_boc5[flag2->cnt] = twbp->p_boc5;
          flag2->i[flag2->cnt] = i;
          flag2->addr[flag2->cnt++] = (long)(&bo_ij->BO);
          if (flag2->cnt == BLKSIZE) {
            buffer_bo_flag2(flag2, system, workspace, vp_boc1, vp_boc2);
            flag2->cnt = 0;
          }    
          // double p_boc1 = system->reax_param.gp.l[0];
          // double p_boc2 = system->reax_param.gp.l[1];
          // double p_boc2_inv = 1.0 / p_boc2;

          // val_j = system->reax_param.sbp[type_j].valency;
          // Deltap_j = workspace->Deltap[j];
          // Deltap_boc_j = workspace->Deltap_boc[j];

          // /* Correction for overcoordination */
          // double exp_p1i = exp(-p_boc1 * Deltap_i);
          // double exp_p2i = exp(-p_boc2 * Deltap_i);
          // double exp_p1j = exp(-p_boc1 * Deltap_j);
          // double exp_p2j = exp(-p_boc2 * Deltap_j);
          // double exp_p2i_p2j_inv = 1.0 / (exp_p2i + exp_p2j);

          // double f2 = exp_p1i + exp_p1j;
          // double f3 = -1.0 * p_boc2_inv * log(0.5 * (exp_p2i  + exp_p2j));
          // double f1 = 0.5 * ((val_i + f2)/(val_i + f2 + f3) +
          //               (val_j + f2)/(val_j + f2 + f3));

          // double temp = f2 + f3;
          // double u1_ij = val_i + temp;
          // double u1_ji = val_j + temp;
          // double u1_ij_inv = 1.0 / u1_ij;
          // double u1_ji_inv = 1.0 / u1_ji;

          // double Cf1A_ij = 0.5 * f3 * (u1_ij_inv*u1_ij_inv + u1_ji_inv*u1_ji_inv);
          // double Cf1B_ij = -0.5 * ((u1_ij - f3) *u1_ij_inv*u1_ij_inv  + (u1_ji - f3) * u1_ji_inv*u1_ji_inv);

          // double Cf1_ij = 0.50 * (-p_boc1 * exp_p1i *u1_ij_inv -
          //           ((val_i+f2) *u1_ij_inv*u1_ij_inv) * (-p_boc1 * exp_p1i + exp_p2i *exp_p2i_p2j_inv) +
          //           -p_boc1 * exp_p1i *u1_ji_inv -
          //           ((val_j+f2) * u1_ji_inv*u1_ji_inv) * (-p_boc1 * exp_p1i + exp_p2i *exp_p2i_p2j_inv));


          // double Cf1_ji = -Cf1A_ij * p_boc1 * exp_p1j + Cf1B_ij * exp_p2j *exp_p2i_p2j_inv; //   / (exp_p2i + exp_p2j);

          // double f1_inv = 1.0 / f1;

          // /* Correction for 1-3 bond orders */
          // double exp_f4 =exp(-(twbp->p_boc4 * SQR(bo_ij->BO) -
          //                   Deltap_boc_i) * twbp->p_boc3 + twbp->p_boc5);
          // double exp_f5 =exp(-(twbp->p_boc4 * SQR(bo_ij->BO) -
          //                   Deltap_boc_j) * twbp->p_boc3 + twbp->p_boc5);
          // double f4 = 1. / (1. + exp_f4);
          // double f5 = 1. / (1. + exp_f5);
          // double f4f5 = f4 * f5;

          // /* Bond Order pages 8-9, derivative of f4 and f5 */
          // double Cf45_ij = -f4 * exp_f4;
          // double Cf45_ji = -f5 * exp_f5;

          // /* Bond Order page 10, derivative of total bond order */
          // double A0_ij = f1 * f4f5;
          // double A1_ij = -2 * twbp->p_boc3 * twbp->p_boc4 * bo_ij->BO *
          //     (Cf45_ij + Cf45_ji);
          // double A2_ij = Cf1_ij / f1 + twbp->p_boc3 * Cf45_ij;
          // double A2_ji = Cf1_ji / f1 + twbp->p_boc3 * Cf45_ji;
          // double A3_ij = A2_ij + Cf1_ij / f1;
          // double A3_ji = A2_ji + Cf1_ji / f1;

          // /* find corrected bond orders and their derivative coef */
          // bo_ij->BO    = bo_ij->BO    * A0_ij;
          // bo_ij->BO_pi = bo_ij->BO_pi * A0_ij *f1;
          // bo_ij->BO_pi2= bo_ij->BO_pi2* A0_ij *f1;
          // bo_ij->BO_s  = bo_ij->BO - (bo_ij->BO_pi + bo_ij->BO_pi2);

          // bo_ij->C1dbo = A0_ij + bo_ij->BO * A1_ij;
          // bo_ij->C2dbo = bo_ij->BO * A2_ij;
          // bo_ij->C3dbo = bo_ij->BO * A2_ji;

          // bo_ij->C1dbopi = f1*f1*f4*f5;
          // bo_ij->C2dbopi = bo_ij->BO_pi * A1_ij;
          // bo_ij->C3dbopi = bo_ij->BO_pi * A3_ij;
          // bo_ij->C4dbopi = bo_ij->BO_pi * A3_ji;

          // bo_ij->C1dbopi2 = f1*f1*f4*f5;
          // bo_ij->C2dbopi2 = bo_ij->BO_pi2 * A1_ij;
          // bo_ij->C3dbopi2 = bo_ij->BO_pi2 * A3_ij;
          // bo_ij->C4dbopi2 = bo_ij->BO_pi2 * A3_ji;

          // workspace->total_bond_order[i] += bo_ij->BO; //now keeps total_BO
        } else if (twbp->v13cor < 0.001 && twbp->ovc >= 0.001) {
          flag3_buffer_t *flag3 = &my_buffers->flag3;
          flag3->val_i[flag3->cnt] = val_i;
          flag3->val_j[flag3->cnt] = system->reax_param.sbp[type_j].valency;
          flag3->Deltap_i[flag3->cnt] = workspace->Deltap[i];
          flag3->Deltap_j[flag3->cnt] = workspace->Deltap[j];
          flag3->BO[flag3->cnt] = bo_ij->BO;
          flag3->BO_pi[flag3->cnt] = bo_ij->BO_pi;
          flag3->BO_pi2[flag3->cnt] = bo_ij->BO_pi2;
          flag3->i[flag3->cnt] = i;
          flag3->addr[flag3->cnt++] = (long)(&bo_ij->BO);
          if (flag3->cnt == BLKSIZE) {
            buffer_bo_flag3(flag3, system, workspace, vp_boc1, vp_boc2);
            flag3->cnt = 0;
          }
          // double p_boc1 = system->reax_param.gp.l[0];
          // double p_boc2 = system->reax_param.gp.l[1];
          // double p_boc2_inv = 1.0 / p_boc2;

          // val_j = system->reax_param.sbp[type_j].valency;
          // Deltap_j = workspace->Deltap[j];
          // Deltap_boc_j = workspace->Deltap_boc[j];

          // /* Correction for overcoordination */
          // double exp_p1i = exp(-p_boc1 * Deltap_i);
          // double exp_p2i = exp(-p_boc2 * Deltap_i);
          // double exp_p1j = exp(-p_boc1 * Deltap_j);
          // double exp_p2j = exp(-p_boc2 * Deltap_j);
          // double exp_p2i_p2j_inv = 1.0 / (exp_p2i + exp_p2j);

          // double f2 = exp_p1i + exp_p1j;
          // double f3 = -1.0 * p_boc2_inv * log(0.5 * (exp_p2i  + exp_p2j));
          // double f1 = 0.5 * ((val_i + f2)/(val_i + f2 + f3) +
          //               (val_j + f2)/(val_j + f2 + f3));

          // double temp = f2 + f3;
          // double u1_ij = val_i + temp;
          // double u1_ji = val_j + temp;
          // double u1_ij_inv = 1.0 / u1_ij;
          // double u1_ji_inv = 1.0 / u1_ji;

          // double Cf1A_ij = 0.5 * f3 * (u1_ij_inv*u1_ij_inv + u1_ji_inv*u1_ji_inv);
          // double Cf1B_ij = -0.5 * ((u1_ij - f3) *u1_ij_inv*u1_ij_inv  + (u1_ji - f3) * u1_ji_inv*u1_ji_inv);

          // double Cf1_ij = 0.50 * (-p_boc1 * exp_p1i *u1_ij_inv -
          //           ((val_i+f2) *u1_ij_inv*u1_ij_inv) * (-p_boc1 * exp_p1i + exp_p2i *exp_p2i_p2j_inv) +
          //           -p_boc1 * exp_p1i *u1_ji_inv -
          //           ((val_j+f2) * u1_ji_inv*u1_ji_inv) * (-p_boc1 * exp_p1i + exp_p2i *exp_p2i_p2j_inv));


          // double Cf1_ji = -Cf1A_ij * p_boc1 * exp_p1j + Cf1B_ij * exp_p2j *exp_p2i_p2j_inv; //   / (exp_p2i + exp_p2j);

          // double f1_inv = 1.0 / f1;

          // double f4, f5, f4f5, Cf45_ij, Cf45_ji;
          // f4 = f5 = f4f5 = 1.0;
          // Cf45_ij = Cf45_ji = 0.0;

          // /* Bond Order page 10, derivative of total bond order */
          // double A0_ij = f1 * f4f5;
          // double A1_ij = -2 * twbp->p_boc3 * twbp->p_boc4 * bo_ij->BO *
          //   (Cf45_ij + Cf45_ji);
          // double A2_ij = Cf1_ij *f1_inv + twbp->p_boc3 * Cf45_ij;
          // double A2_ji = Cf1_ji *f1_inv + twbp->p_boc3 * Cf45_ji;
          // double A3_ij = A2_ij + Cf1_ij *f1_inv;
          // double A3_ji = A2_ji + Cf1_ji *f1_inv;

          // /* find corrected bond orders and their derivative coef */
          // bo_ij->BO    = bo_ij->BO    * A0_ij;
          // bo_ij->BO_pi = bo_ij->BO_pi * A0_ij *f1;
          // bo_ij->BO_pi2= bo_ij->BO_pi2* A0_ij *f1;
          // bo_ij->BO_s  = bo_ij->BO - (bo_ij->BO_pi + bo_ij->BO_pi2);

          // bo_ij->C1dbo = A0_ij + bo_ij->BO * A1_ij;
          // bo_ij->C2dbo = bo_ij->BO * A2_ij;
          // bo_ij->C3dbo = bo_ij->BO * A2_ji;

          // bo_ij->C1dbopi = f1*f1*f4*f5;
          // bo_ij->C2dbopi = bo_ij->BO_pi * A1_ij;
          // bo_ij->C3dbopi = bo_ij->BO_pi * A3_ij;
          // bo_ij->C4dbopi = bo_ij->BO_pi * A3_ji;

          // bo_ij->C1dbopi2 = f1*f1*f4*f5;
          // bo_ij->C2dbopi2 = bo_ij->BO_pi2 * A1_ij;
          // bo_ij->C3dbopi2 = bo_ij->BO_pi2 * A3_ij;
          // bo_ij->C4dbopi2 = bo_ij->BO_pi2 * A3_ji;

          // workspace->total_bond_order[i] += bo_ij->BO; //now keeps total_BO
        } else {
          bo_ij->C1dbo = 1.000000;
          bo_ij->C2dbo = 0.000000;
          bo_ij->C3dbo = 0.000000;

          bo_ij->C1dbopi = 1.000000;
          bo_ij->C2dbopi = 0.000000;
          bo_ij->C3dbopi = 0.000000;
          bo_ij->C4dbopi = 0.000000;

          bo_ij->C1dbopi2 = 1.000000;
          bo_ij->C2dbopi2 = 0.000000;
          bo_ij->C3dbopi2 = 0.000000;
          bo_ij->C4dbopi2 = 0.000000;
                
          workspace->total_bond_order[i] += bo_ij->BO;
        }
      }
    }

    if (my_buffers->flag1.cnt > 0) {
      buffer_bo_flag1(&my_buffers->flag1, system, workspace);
    }
    if (my_buffers->flag2.cnt > 0) {
      buffer_bo_flag2(&my_buffers->flag2, system, workspace, vp_boc1, vp_boc2);
    }
    if (my_buffers->flag3.cnt > 0) {
      buffer_bo_flag3(&my_buffers->flag3, system, workspace, vp_boc1, vp_boc2);
    }
  }

    p_lp1 = system->reax_param.gp.l[15];

    // for (int xx = 0; xx < system->N; ++xx) {
    //   printf("bond order: %d %f\n", xx, workspace->total_bond_order[xx]);
    // }

    #pragma omp parallel for
    for (int j = 0; j < system->N; ++j) {
      int type_j = system->my_atoms[j].type;
      single_body_parameters *sbp_j = &(system->reax_param.sbp[type_j]);
  
      workspace->Delta[j] = workspace->total_bond_order[j] - sbp_j->valency;
      workspace->Delta_e[j] = workspace->total_bond_order[j] - sbp_j->valency_e;
      workspace->Delta_boc[j] = workspace->total_bond_order[j] - sbp_j->valency_boc;
      workspace->Delta_val[j] = workspace->total_bond_order[j] - sbp_j->valency_val;
  
      workspace->vlpex[j] = workspace->Delta_e[j] - 2.0 * (int)(workspace->Delta_e[j] * 0.5);
      double explp1 = exp(-p_lp1 * SQR(2.0 + workspace->vlpex[j]));
      workspace->nlp[j] = explp1 - (int)(workspace->Delta_e[j] * 0.5);
      workspace->Delta_lp[j] = sbp_j->nlp_opt - workspace->nlp[j];
      workspace->Clp[j] = 2.0 * p_lp1 * explp1 * (2.0 + workspace->vlpex[j]);
      workspace->dDelta_lp[j] = workspace->Clp[j];
    
      if (sbp_j->mass > 21.0) {
        workspace->nlp_temp[j] = 0.5 * (sbp_j->valency_e - sbp_j->valency);
        workspace->Delta_lp_temp[j] = sbp_j->nlp_opt - workspace->nlp_temp[j];
        workspace->dDelta_lp_temp[j] = 0.;
      } else {
        workspace->nlp_temp[j] = workspace->nlp[j];
        workspace->Delta_lp_temp[j] = sbp_j->nlp_opt - workspace->nlp_temp[j];
        workspace->dDelta_lp_temp[j] = workspace->Clp[j];
      }
    }

  }
    
}