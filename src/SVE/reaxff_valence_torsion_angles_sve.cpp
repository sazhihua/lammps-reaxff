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

#include <cmath>
#include "pair.h"
#include "error.h"
#include <omp.h> 

namespace ReaxFFSVE 
{
/******************merge*****************************/
void Valence_and_Torsion_Angles(reax_system *system, control_params *control, simulation_data *data,
                      storage *workspace, reax_list **lists)
{
  int nthr = omp_get_max_threads();

  const double p_tor2 = system->reax_param.gp.l[23];
  const double p_tor3 = system->reax_param.gp.l[24];
  const double p_tor4 = system->reax_param.gp.l[25];
  const double p_cot2 = system->reax_param.gp.l[27];
  reax_list *bonds = (*lists) + BONDS;

  double total_e_ang = 0.0;
  double total_e_pen = 0.0;
  double total_e_coa = 0.0;
  double total_e_tor = 0.0;
  double total_e_con = 0.0;

#pragma omp parallel num_threads(nthr) reduction(+: total_e_ang, total_e_pen, total_e_coa, total_e_tor, total_e_con)
  {
    int ithr = omp_get_thread_num();
    int i, j, k, l, pi, /*pj, */pk, pl, pij, plk, natoms, pw, w, t;
    int type_i, type_j, type_k, type_l, type_w;
    int start_j, end_j, start_k, end_k;
    int start_pj, end_pj;

    double Delta_j, Delta_k;
    double r_ij, r_jk, r_kl, r_li;
    double BOA_ij, BOA_jk, BOA_kl;

    double exp_tor2_ij, exp_tor2_jk, exp_tor2_kl;
    double exp_tor1, exp_tor3_DjDk, exp_tor4_DjDk, exp_tor34_inv;
    double exp_cot2_jk, exp_cot2_ij, exp_cot2_kl;
    double fn10, f11_DjDk, dfn11, fn12;
    double theta_ijk, theta_jkl;
    double cos_theta_jkl, cos_theta_ijk;
    double sin_ijk, sin_jkl;
    double cos_ijk, cos_jkl;
    double tan_ijk_i, tan_jkl_i;
    double omega, cos_omega, cos2omega, cos3omega;
    rvec dcos_omega_di, dcos_omega_dj, dcos_omega_dk, dcos_omega_dl;
    double CV, cmn, CEtors1, CEtors2, CEtors3, CEtors4;
    double CEtors5, CEtors6, CEtors7, CEtors8, CEtors9;
    double Cconj, CEconj1, CEconj2, CEconj3;
    double CEconj4, CEconj5, CEconj6;
    double e_tor, e_con;
    rvec dvec_li;
    four_body_header *fbh;
    four_body_parameters *fbp;
    bond_data *pbond_ij, *pbond_jk, *pbond_jt, *pbond_kl; //, *pbond_pj;
    bond_order_data *bo_ij, *bo_jk, *bo_jt, *bo_kl;
    //three_body_interaction_data *p_ijk, *p_jkl;
    //three_body_interaction_data thb_intrs_data, thb_intrs_data_ijk;
    
    int p_ijk_thb, p_ijk_pthb;
    double p_ijk_theta, p_ijk_cos_theta;
    rvec p_ijk_dcos_di, p_ijk_dcos_dj, p_ijk_dcos_dk;
    
    int p_jkl_thb, p_jkl_pthb;
    double p_jkl_theta, p_jkl_cos_theta;
    rvec p_jkl_dcos_di, p_jkl_dcos_dj, p_jkl_dcos_dk;

    double eng_tmp;

    natoms = system->n;

    //valence:
    int cnt;
    double temp, temp_bo_jt, pBOjt7;
    double p_val1, p_val2, p_val3, p_val4, p_val5;
    double p_val6, p_val7, p_val8, p_val9, p_val10;
    double p_pen1, p_pen2, p_pen3, p_pen4;
    double p_coa1, p_coa2, p_coa3, p_coa4;
    double trm8, expval6, expval7, expval2theta, expval12theta, exp3ij, exp3jk;
    double exp_pen2ij, exp_pen2jk, exp_pen3, exp_pen4, trm_pen34, exp_coa2;
    double dSBO1, dSBO2, SBO, SBO2, CSBO2, SBOp, prod_SBO, vlpadj;
    double CEval1, CEval2, CEval3, CEval4, CEval5, CEval6, CEval7, CEval8;
    double CEpen1, CEpen2, CEpen3;
    double e_ang, e_coa, e_pen;
    double CEcoa1, CEcoa2, CEcoa3, CEcoa4, CEcoa5;
    double Cf7ij, Cf7jk, Cf8j, Cf9j;
    double f7_ij, f7_jk, f8_Dj, f9_Dj;
    double Ctheta_0, theta_0, theta_00, theta, cos_theta, sin_theta_ijk;
    three_body_header *thbh;
    three_body_parameters *thbp;
 

    /* global parameters used in these calculations */
    p_val6 = system->reax_param.gp.l[14];
    p_val8 = system->reax_param.gp.l[33];
    p_val9 = system->reax_param.gp.l[16];
    p_val10 = system->reax_param.gp.l[17];

    p_pen2 = system->reax_param.gp.l[19];
    p_pen3 = system->reax_param.gp.l[20];
    p_pen4 = system->reax_param.gp.l[21];

    p_coa2 = system->reax_param.gp.l[2];
    p_coa3 = system->reax_param.gp.l[38];
    p_coa4 = system->reax_param.gp.l[30];


    double thb_cut = control->thb_cut;
    double thb_cutsq = control->thb_cutsq;
    double min_sine_inv = 1.0 / MIN_SINE;
    double exp_p_val10, exp_coa2_inv;
    double trm8_inv, trm_pen34_inv;
    double sin_ijk_inv, sin_jkl_inv;
    double r_ij_inv,  r_jk_inv, r_kl_inv;
 
    double sqr_d_ji, sqr_d_jk;
    double inv_dists, inv_dists3;
    double dot_dvecs, Cdot_inv3;

    double (*my_f)[3] = (double(*)[3])workspace->forceTmp[ithr][0];
    double *my_C = workspace->CdDeltaTmp[ithr];
    memset(my_f, 0.0, sizeof(double) * system->N * 3); // first use, so zeroed

    #pragma omp for schedule(dynamic, 64)
    for (j = 0; j < system->n; ++j)
    {
      type_j = system->my_atoms[j].type;
      if (type_j < 0) continue;
      Delta_j = workspace->Delta_boc[j];
      start_j = Start_Index(j, bonds);
      end_j = End_Index(j, bonds);

      if(end_j <= start_j) continue;

      p_val3 = system->reax_param.sbp[type_j].p_val3;
      p_val5 = system->reax_param.sbp[type_j].p_val5;

      SBOp = 0, prod_SBO = 1;
      for (t = start_j; t < end_j; ++t) 
      {
        bo_jt = &(bonds->select.bond_list[t].bo_data);
        SBOp += (bo_jt->BO_pi + bo_jt->BO_pi2);
        temp = SQR(bo_jt->BO);
        temp *= temp;
        temp *= temp;
        prod_SBO *= exp(-temp);
      }

      if (workspace->vlpex[j] >= 0) 
      {
        vlpadj = 0;
        dSBO2 = prod_SBO - 1;
      } 
      else {
        vlpadj = workspace->nlp[j];
        dSBO2 = (prod_SBO - 1) * (1 - p_val8 * workspace->dDelta_lp[j]);
      }

      SBO = SBOp + (1 - prod_SBO) * (-workspace->Delta_boc[j] - p_val8 * vlpadj);
      dSBO1 = -8 * prod_SBO * (workspace->Delta_boc[j] + p_val8 * vlpadj);

      if (SBO <= 0)
        SBO2 = 0, CSBO2 = 0;
      else if (SBO > 0 && SBO <= 1) {
        SBO2 = pow(SBO, p_val9);
        CSBO2 = p_val9 * pow(SBO, p_val9 - 1);
      }
      else if (SBO > 1 && SBO < 2) {
        SBO2 = 2 - pow(2-SBO, p_val9);
        CSBO2 = p_val9 * pow(2 - SBO, p_val9 - 1);
      }
      else
        SBO2 = 2, CSBO2 = 0;

      expval6 = exp(p_val6 * workspace->Delta_boc[j]);


      for (pk = start_j; pk < end_j; ++pk) 
      {

        pbond_jk = &(bonds->select.bond_list[pk]);
        bo_jk = &(pbond_jk->bo_data);
        BOA_jk = bo_jk->BO - thb_cut;
        k = pbond_jk->nbr;


        start_k = Start_Index(k, bonds);
        end_k = End_Index(k, bonds);

        if(end_k > start_k && BOA_jk > 0.0)
        { 
          type_k = system->my_atoms[k].type;
          Delta_k = workspace->Delta_boc[k];
          r_jk = pbond_jk->d;

          exp_tor2_jk = exp(-p_tor2 * BOA_jk);
          exp_cot2_jk = exp(-p_cot2 * SQR(BOA_jk - 1.5));
          exp_tor3_DjDk = exp(-p_tor3 * (Delta_j + Delta_k));
          exp_tor4_DjDk = exp(p_tor4  * (Delta_j + Delta_k));
          exp_tor34_inv = 1.0 / (1.0 + exp_tor3_DjDk + exp_tor4_DjDk);
          f11_DjDk = (2.0 + exp_tor3_DjDk) * exp_tor34_inv;


          for (pi = start_j; pi < end_j; ++pi) 
          {
            pbond_ij  = &(bonds->select.bond_list[pi]);
            bo_ij     = &(pbond_ij->bo_data);
            BOA_ij    = bo_ij->BO - thb_cut;
            i         = pbond_ij->nbr;
            type_i    = system->my_atoms[i].type;

            if (bo_ij->BO <= thb_cut || pi==pk) continue;

            cos_theta_ijk = rvec_Dot(pbond_jk->dvec, pbond_ij->dvec) / (pbond_jk->d * pbond_ij->d);
            if (cos_theta_ijk > 1.) cos_theta_ijk  = 1.0;
            if (cos_theta_ijk < -1.) cos_theta_ijk  = -1.0;
            theta_ijk = acos(cos_theta_ijk);

            sqr_d_ji = SQR(pbond_jk->d);
            sqr_d_jk = SQR(pbond_ij->d);
            inv_dists = 1.0 / (pbond_jk->d * pbond_ij->d);
            inv_dists3 = CUBE(inv_dists);
            dot_dvecs = rvec_Dot(pbond_jk->dvec, pbond_ij->dvec);
            Cdot_inv3 = dot_dvecs * inv_dists3;

            for (t = 0; t < 3; ++t) {
              p_ijk_dcos_di[t] = pbond_ij->dvec[t] * inv_dists -
                Cdot_inv3 * sqr_d_jk * pbond_jk->dvec[t];
              p_ijk_dcos_dj[t] = -(pbond_ij->dvec[t] + pbond_jk->dvec[t]) * inv_dists +
                Cdot_inv3 * (sqr_d_jk * pbond_jk->dvec[t] + sqr_d_ji * pbond_ij->dvec[t]);
              p_ijk_dcos_dk[t] = pbond_jk->dvec[t] * inv_dists -
                Cdot_inv3 * sqr_d_ji * pbond_ij->dvec[t];
            }


            p_ijk_thb = i;
            p_ijk_pthb = pi;
            p_ijk_theta = theta_ijk;

            sin_theta_ijk = sin(theta_ijk);
            if (sin_theta_ijk < 1.0e-5) sin_theta_ijk = 1.0e-5;

            if ((pi > pk) && (bo_ij->BO > thb_cut) && 
                (bo_jk->BO > thb_cut) && (bo_ij->BO * bo_jk->BO > thb_cutsq)) 
            {
              thbh = &(system->reax_param.thbp[type_k][type_j][type_i]);

              for (cnt = 0; cnt < thbh->cnt; ++cnt) 
              {
                if (fabs(thbh->prm[cnt].p_val1) > 0.001) 
                {
                  thbp = &(thbh->prm[cnt]);

                  /* ANGLE ENERGY */
                  p_val1 = thbp->p_val1;
                  p_val2 = thbp->p_val2;
                  p_val4 = thbp->p_val4;
                  p_val7 = thbp->p_val7;
                  theta_00 = thbp->theta_00;

                  exp3jk = exp(-p_val3 * pow(BOA_jk, p_val4));
                  f7_jk = 1.0 - exp3jk;
                  Cf7jk = p_val3 * p_val4 * pow(BOA_jk, p_val4 - 1.0) * exp3jk;

                  exp3ij = exp(-p_val3 * pow(BOA_ij, p_val4));
                  f7_ij = 1.0 - exp3ij;
                  Cf7ij = p_val3 * p_val4 * pow(BOA_ij, p_val4 - 1.0) * exp3ij;

                  expval7 = exp(-p_val7 * workspace->Delta_boc[j]);
                  trm8 = 1.0 + expval6 + expval7;
                  exp_p_val10 = exp(-p_val10 * (2.0 - SBO2));
                  trm8_inv = 1.0 / trm8;
                  theta_0 = 180.0 - theta_00 * (1.0 - exp_p_val10);
                  f8_Dj = p_val5 - ((p_val5 - 1.0) * (2.0 + expval6) * trm8_inv);
                  Cf8j = ((1.0 - p_val5) *trm8_inv*trm8_inv) * (p_val6 * expval6 * trm8 -
                      (2.0 + expval6) * (p_val6*expval6 - p_val7*expval7));
                  
                  theta_0 = DEG2RAD(theta_0);

                  expval2theta  = exp(-p_val2 * SQR(theta_0 - theta_ijk));
                  if (p_val1 >= 0)
                    expval12theta = p_val1 * (1.0 - expval2theta);
                  else // To avoid linear Me-H-Me angles (6/6/06)
                    expval12theta = p_val1 * -expval2theta;

                  CEval1 = Cf7jk * f7_ij * f8_Dj * expval12theta;
                  CEval2 = Cf7ij * f7_jk * f8_Dj * expval12theta;
                  CEval3 = Cf8j  * f7_ij * f7_jk * expval12theta;
                  CEval4 = -2.0 * p_val1 * p_val2 * f7_jk * f7_ij * f8_Dj *
                            expval2theta * (theta_0 - theta_ijk);

                  Ctheta_0 = p_val10 * DEG2RAD(theta_00) * exp_p_val10;

                  CEval5 = -CEval4 * Ctheta_0 * CSBO2;
                  CEval6 = CEval5 * dSBO1;
                  CEval7 = CEval5 * dSBO2;
                  CEval8 = -CEval4 / sin_theta_ijk;

                  // data->my_en.e_ang += e_ang = f7_jk * f7_ij * f8_Dj * expval12theta;
                  // eng_virial[0] += e_ang = f7_jk * f7_ij * f8_Dj * expval12theta;
                  total_e_ang += e_ang = f7_jk * f7_ij * f8_Dj * expval12theta;
                  /* END ANGLE ENERGY*/

                  /* PENALTY ENERGY */
                  p_pen1 = thbp->p_pen1;
                  
                  exp_pen3 = exp(-p_pen3 * workspace->Delta[j]);
                  exp_pen4 = exp( p_pen4 * workspace->Delta[j]);
                  exp_pen2jk = exp(-p_pen2 * SQR(BOA_jk - 2.0));
                  trm_pen34 = 1.0 + exp_pen3 + exp_pen4;
                  exp_pen2ij = exp(-p_pen2 * SQR(BOA_ij - 2.0));
                  trm_pen34_inv = 1.0 / trm_pen34;
                  f9_Dj = (2.0 + exp_pen3) * trm_pen34_inv;
                  Cf9j = (-p_pen3 * exp_pen3 * trm_pen34 - (2.0 + exp_pen3) * (-p_pen3 * exp_pen3 +
                           p_pen4 * exp_pen4)) *trm_pen34_inv * trm_pen34_inv;

                  // data->my_en.e_pen += e_pen = p_pen1 * f9_Dj * exp_pen2jk * exp_pen2ij;
                  // eng_virial[1] += e_pen = p_pen1 * f9_Dj * exp_pen2jk * exp_pen2ij;
                  total_e_pen += e_pen = p_pen1 * f9_Dj * exp_pen2jk * exp_pen2ij;

                  CEpen1 = e_pen * Cf9j / f9_Dj;
                  temp   = -2.0 * p_pen2 * e_pen;
                  CEpen2 = temp * (BOA_jk - 2.0);
                  CEpen3 = temp * (BOA_ij - 2.0);
                  /* END PENALTY ENERGY */

                  /* COALITION ENERGY */
                  p_coa1 = thbp->p_coa1;
                  
                  exp_coa2 = exp(p_coa2 * workspace->Delta_val[j]);
                  exp_coa2_inv = 1.0 / (1. + exp_coa2);
                  // data->my_en.e_coa += e_coa = p_coa1 * exp_coa2_inv *
                  // eng_virial[2] += e_coa = p_coa1 * exp_coa2_inv *
                  total_e_coa += e_coa = p_coa1 * exp_coa2_inv *
                    exp(-p_coa3 * SQR(workspace->total_bond_order[k]-BOA_jk)) *
                    exp(-p_coa3 * SQR(workspace->total_bond_order[i]-BOA_ij)) *
                    exp(-p_coa4 * SQR(BOA_jk - 1.5)) *
                    exp(-p_coa4 * SQR(BOA_ij - 1.5));


                  CEcoa1 = -2 * p_coa4 * (BOA_jk - 1.5) * e_coa;
                  CEcoa2 = -2 * p_coa4 * (BOA_ij - 1.5) * e_coa;
                  CEcoa3 = -p_coa2 * exp_coa2 * e_coa * exp_coa2_inv;
                  CEcoa4 = -2 * p_coa3 * (workspace->total_bond_order[k]-BOA_jk) * e_coa;
                  CEcoa5 = -2 * p_coa3 * (workspace->total_bond_order[i]-BOA_ij) * e_coa;
              
              

                  /* END COALITION ENERGY */

                  /* FORCES */
                  bo_jk->Cdbo += (CEval1 + CEpen2 + (CEcoa1 - CEcoa4));
                  bo_ij->Cdbo += (CEval2 + CEpen3 + (CEcoa2 - CEcoa5));
                  // workspace->CdDelta[j] += ((CEval3 + CEval7) + CEpen1 + CEcoa3);
                  // workspace->CdDelta[k] += CEcoa4;
                  // workspace->CdDelta[i] += CEcoa5;
                  my_C[j] += ((CEval3 + CEval7) + CEpen1 + CEcoa3);
                  my_C[k] += CEcoa4;
                  my_C[i] += CEcoa5;

                  for (t = start_j; t < end_j; ++t) 
                  {
                    pbond_jt = &(bonds->select.bond_list[t]);
                    bo_jt = &(pbond_jt->bo_data);
                    temp_bo_jt = bo_jt->BO;
                    temp = CUBE(temp_bo_jt);
                    pBOjt7 = temp * temp * temp_bo_jt;

                    bo_jt->Cdbo += (CEval6 * pBOjt7);
                    bo_jt->Cdbopi += CEval5;
                    bo_jt->Cdbopi2 += CEval5;
                  }

                  // rvec_ScaledAdd(workspace->f[k], CEval8, p_ijk_dcos_di);
                  // rvec_ScaledAdd(workspace->f[j], CEval8, p_ijk_dcos_dj);
                  // rvec_ScaledAdd(workspace->f[i], CEval8, p_ijk_dcos_dk);
                  rvec_ScaledAdd(my_f[k], CEval8, p_ijk_dcos_di);
                  rvec_ScaledAdd(my_f[j], CEval8, p_ijk_dcos_dj);
                  rvec_ScaledAdd(my_f[i], CEval8, p_ijk_dcos_dk);

                  /* tally energy */
                  // if (system->pair_ptr->eflag_either) 
                  // {
                  //   eng_tmp = e_ang + e_pen + e_coa;
                  //   // system->pair_ptr->eng_vdwl += eng_tmp;
                  //   // eng_virial[5] += eng_tmp;
                  //   total_e_vdwl += eng_tmp;
                  // }

                }
              }
            }//if

            ////Torsion:
            if (system->my_atoms[j].orig_id > system->my_atoms[k].orig_id)
              continue;
            if (system->my_atoms[j].orig_id == system->my_atoms[k].orig_id) 
            {
              if (system->my_atoms[k].x[2] <  system->my_atoms[j].x[2]) continue;
              if (system->my_atoms[k].x[2] == system->my_atoms[j].x[2] &&
                  system->my_atoms[k].x[1] <  system->my_atoms[j].x[1]) continue;
              if (system->my_atoms[k].x[2] == system->my_atoms[j].x[2] &&
                  system->my_atoms[k].x[1] == system->my_atoms[j].x[1] &&
                  system->my_atoms[k].x[0] <  system->my_atoms[j].x[0]) continue;
            }

            r_ij = pbond_ij->d;
            sin_ijk = sin_theta_ijk; //sin(theta_ijk);
            cos_ijk = cos(theta_ijk);
            sin_ijk_inv = 1.0 / sin_ijk;

            if (sin_ijk >= 0 && sin_ijk <= MIN_SINE)
              tan_ijk_i = cos_ijk * min_sine_inv; // / MIN_SINE;
            else if (sin_ijk <= 0 && sin_ijk >= -MIN_SINE)
              tan_ijk_i = cos_ijk * -min_sine_inv; // / -MIN_SINE;
            else tan_ijk_i = cos_ijk * sin_ijk_inv;

            exp_tor2_ij = exp(-p_tor2 * BOA_ij);
            exp_cot2_ij = exp(-p_cot2 * SQR(BOA_ij -1.5));

            for (pl = start_k; pl < end_k; ++pl) 
            {
              pbond_kl = &(bonds->select.bond_list[pl]);
              bo_kl = &(pbond_kl->bo_data);
              BOA_kl = bo_kl->BO - thb_cut;
              l = pbond_kl->nbr;
              if (l == j) continue;
              type_l = system->my_atoms[l].type;
              fbh = &(system->reax_param.fbp[type_i][type_j][type_k][type_l]);

              if (i != l && fbh->cnt && bo_kl->BO > thb_cut  &&
                   bo_ij->BO * bo_jk->BO * bo_kl->BO > thb_cut) 
              {

                cos_theta_jkl = -rvec_Dot(pbond_jk->dvec,pbond_kl->dvec) / (pbond_jk->d * pbond_kl->d);
                if (cos_theta_jkl > 1.) cos_theta_jkl  = 1.0;
                if (cos_theta_jkl < -1.) cos_theta_jkl  = -1.0;
                theta_jkl = acos(cos_theta_jkl);

                sqr_d_ji = SQR(pbond_jk->d);
                sqr_d_jk = SQR(pbond_kl->d);
                inv_dists = 1.0 / (pbond_jk->d * pbond_kl->d);
                inv_dists3 = CUBE(inv_dists);
                dot_dvecs = -rvec_Dot(pbond_jk->dvec, pbond_kl->dvec);
                Cdot_inv3 = dot_dvecs * inv_dists3;
                for (t = 0; t < 3; ++t) 
                {
                  p_jkl_dcos_di[t] = pbond_kl->dvec[t] * inv_dists -
                    Cdot_inv3 * sqr_d_jk * (-pbond_jk->dvec[t]);
                  p_jkl_dcos_dj[t] = -(pbond_kl->dvec[t] + (-pbond_jk->dvec[t])) * inv_dists +
                    Cdot_inv3 * (sqr_d_jk * (-pbond_jk->dvec[t]) + sqr_d_ji * pbond_kl->dvec[t]);
                  p_jkl_dcos_dk[t] = (-pbond_jk->dvec[t]) * inv_dists -
                    Cdot_inv3 * sqr_d_ji * pbond_kl->dvec[t];
                }

                p_jkl_thb = l;
                p_jkl_pthb = pl;
                p_jkl_theta = theta_jkl;

                fbp = &(system->reax_param.fbp[type_i][type_j][type_k][type_l].prm[0]);
              
                r_kl = pbond_kl->d;

                sin_jkl = sin(theta_jkl);
                cos_jkl = cos(theta_jkl);
                sin_jkl_inv = 1.0 / sin_jkl;

                if (sin_jkl >= 0 && sin_jkl <= MIN_SINE)
                  tan_jkl_i = cos_jkl * min_sine_inv; /// MIN_SINE;
                else if (sin_jkl <= 0 && sin_jkl >= -MIN_SINE)
                  tan_jkl_i = cos_jkl * -min_sine_inv; // / -MIN_SINE;
                else tan_jkl_i = cos_jkl *sin_jkl_inv;

                rvec_ScaledSum(dvec_li, 1., system->my_atoms[i].x, -1., system->my_atoms[l].x);
                r_li = rvec_Norm(dvec_li);


                /* omega and its derivative */
                double unnorm_cos_omega, unnorm_sin_omega, omega;
                double htra, htrb, htrc, hthd, hthe, hnra, hnrc, hnhd, hnhe;
                double arg, poem, tel;
                rvec cross_jk_kl;
                
                /* omega */
                unnorm_cos_omega = -rvec_Dot(pbond_ij->dvec, pbond_jk->dvec) * rvec_Dot(pbond_jk->dvec, 
                                pbond_kl->dvec) + SQR(r_jk) *  rvec_Dot(pbond_ij->dvec, pbond_kl->dvec);

                rvec_Cross(cross_jk_kl, pbond_jk->dvec, pbond_kl->dvec);
                unnorm_sin_omega = -r_jk * rvec_Dot(pbond_ij->dvec, cross_jk_kl);

                omega = atan2(unnorm_sin_omega, unnorm_cos_omega);

                htra = r_ij + cos_ijk * (r_kl * cos_jkl - r_jk);
                htrb = r_jk - r_ij * cos_ijk - r_kl * cos_jkl;
                htrc = r_kl + cos_jkl * (r_ij * cos_ijk - r_jk);
                hthd = r_ij * sin_ijk * (r_jk - r_kl * cos_jkl);
                hthe = r_kl * sin_jkl * (r_jk - r_ij * cos_ijk);
                hnra = r_kl * sin_ijk * sin_jkl;
                hnrc = r_ij * sin_ijk * sin_jkl;
                hnhd = r_ij * r_kl * cos_ijk * sin_jkl;
                hnhe = r_ij * r_kl * sin_ijk * cos_jkl;

                tel  = SQR(r_ij) + SQR(r_jk) + SQR(r_kl) - SQR(r_li) -
                        2.0 * (r_ij * r_jk * cos_ijk - 
                        r_ij * r_kl * cos_ijk * cos_jkl + r_jk * r_kl * cos_jkl);

                poem = 2.0 * r_ij * r_kl * sin_ijk * sin_jkl;

                arg  = tel / poem;

                if (arg >  1.0) arg =  1.0;
                if (arg < -1.0) arg = -1.0;

                double poem_inv2 = 2.0 / poem;
                r_ij_inv = 1.0 / r_ij;
                r_jk_inv = 1.0 / r_jk;
                r_kl_inv = 1.0 / r_kl;

                // dcos_omega_di
                rvec_ScaledSum(dcos_omega_di, (htra-arg*hnra)*r_ij_inv, pbond_ij->dvec, -1., dvec_li);
                rvec_ScaledAdd(dcos_omega_di,-(hthd-arg*hnhd)*sin_ijk_inv, p_ijk_dcos_dk);
                rvec_Scale(dcos_omega_di, poem_inv2, dcos_omega_di);

                // dcos_omega_dj
                rvec_ScaledSum(dcos_omega_dj,-(htra-arg*hnra)*r_ij_inv, pbond_ij->dvec, -htrb *r_jk_inv, pbond_jk->dvec);
                rvec_ScaledAdd(dcos_omega_dj,-(hthd-arg*hnhd)*sin_ijk_inv, p_ijk_dcos_dj);
                rvec_ScaledAdd(dcos_omega_dj,-(hthe-arg*hnhe)*sin_jkl_inv, p_jkl_dcos_di);
                rvec_Scale(dcos_omega_dj, poem_inv2, dcos_omega_dj);

                // dcos_omega_dk
                rvec_ScaledSum(dcos_omega_dk,-(htrc-arg*hnrc)*r_kl_inv, pbond_kl->dvec, htrb *r_jk_inv, pbond_jk->dvec);
                rvec_ScaledAdd(dcos_omega_dk,-(hthd-arg*hnhd)*sin_ijk_inv, p_ijk_dcos_di);
                rvec_ScaledAdd(dcos_omega_dk,-(hthe-arg*hnhe)*sin_jkl_inv, p_jkl_dcos_dj);
                rvec_Scale(dcos_omega_dk, poem_inv2, dcos_omega_dk);

                // dcos_omega_dl
                rvec_ScaledSum(dcos_omega_dl, (htrc-arg*hnrc)*r_kl_inv, pbond_kl->dvec, 1., dvec_li);
                rvec_ScaledAdd(dcos_omega_dl,-(hthe-arg*hnhe)*sin_jkl_inv, p_jkl_dcos_dk);
                rvec_Scale(dcos_omega_dl, poem_inv2, dcos_omega_dl);


                cos_omega = cos(omega);
                cos2omega = cos(2. * omega);
                cos3omega = cos(3. * omega);
                /* end omega calculations */

                /* torsion energy */
                exp_tor1 = exp(fbp->p_tor1 * SQR(2.0 - bo_jk->BO_pi - f11_DjDk));
                exp_tor2_kl = exp(-p_tor2 * BOA_kl);
                exp_cot2_kl = exp(-p_cot2 * SQR(BOA_kl - 1.5));
                fn10 = (1.0 - exp_tor2_ij) * (1.0 - exp_tor2_jk) * (1.0 - exp_tor2_kl);

                CV = 0.5 * (fbp->V1 * (1.0 + cos_omega) +
                             fbp->V2 * exp_tor1 * (1.0 - cos2omega) +
                             fbp->V3 * (1.0 + cos3omega));

                // data->my_en.e_tor += e_tor = fn10 * sin_ijk * sin_jkl * CV;
                // eng_virial[3] += e_tor = fn10 * sin_ijk * sin_jkl * CV;
                total_e_tor += e_tor = fn10 * sin_ijk * sin_jkl * CV;

                dfn11 = (-p_tor3 * exp_tor3_DjDk +
                         (p_tor3 * exp_tor3_DjDk - p_tor4 * exp_tor4_DjDk) *
                         (2.0 + exp_tor3_DjDk) * exp_tor34_inv) * exp_tor34_inv;

                CEtors1 = sin_ijk * sin_jkl * CV;

                CEtors2 = -fn10 * 2.0 * fbp->p_tor1 * fbp->V2 * exp_tor1 *
                  (2.0 - bo_jk->BO_pi - f11_DjDk) * (1.0 - SQR(cos_omega)) *
                  sin_ijk * sin_jkl;
                CEtors3 = CEtors2 * dfn11;

                CEtors4 = CEtors1 * p_tor2 * exp_tor2_ij *
                  (1.0 - exp_tor2_jk) * (1.0 - exp_tor2_kl);
                CEtors5 = CEtors1 * p_tor2 *
                  (1.0 - exp_tor2_ij) * exp_tor2_jk * (1.0 - exp_tor2_kl);
                CEtors6 = CEtors1 * p_tor2 *
                  (1.0 - exp_tor2_ij) * (1.0 - exp_tor2_jk) * exp_tor2_kl;

                cmn = -fn10 * CV;
                CEtors7 = cmn * sin_jkl * tan_ijk_i;
                CEtors8 = cmn * sin_ijk * tan_jkl_i;

                CEtors9 = fn10 * sin_ijk * sin_jkl *
                  (0.5 * fbp->V1 - 2.0 * fbp->V2 * exp_tor1 * cos_omega +
                   1.5 * fbp->V3 * (cos2omega + 2.0 * SQR(cos_omega)));
                /* end  of torsion energy */

                /* 4-body conjugation energy */
                fn12 = exp_cot2_ij * exp_cot2_jk * exp_cot2_kl;
                // data->my_en.e_con += e_con = fbp->p_cot1 * fn12 *
                // eng_virial[4] += e_con = fbp->p_cot1 * fn12 *
                total_e_con += e_con = fbp->p_cot1 * fn12 *
                  (1.0 + (SQR(cos_omega) - 1.0) * sin_ijk * sin_jkl);

                Cconj = -2.0 * fn12 * fbp->p_cot1 * p_cot2 *
                  (1.0 + (SQR(cos_omega) - 1.0) * sin_ijk * sin_jkl);

                CEconj1 = Cconj * (BOA_ij - 1.5e0);
                CEconj2 = Cconj * (BOA_jk - 1.5e0);
                CEconj3 = Cconj * (BOA_kl - 1.5e0);

                CEconj4 = -fbp->p_cot1 * fn12 * (SQR(cos_omega) - 1.0) * sin_jkl * tan_ijk_i;
                CEconj5 = -fbp->p_cot1 * fn12 * (SQR(cos_omega) - 1.0) * sin_ijk * tan_jkl_i;
                CEconj6 = 2.0 * fbp->p_cot1 * fn12 * cos_omega * sin_ijk * sin_jkl;
                /* end 4-body conjugation energy */

                /* forces */
                bo_jk->Cdbopi += CEtors2;
                // workspace->CdDelta[j] += CEtors3;
                // workspace->CdDelta[k] += CEtors3;
                my_C[j] += CEtors3; 
                my_C[k] += CEtors3;
                bo_ij->Cdbo += (CEtors4 + CEconj1);
                bo_jk->Cdbo += (CEtors5 + CEconj2);
                bo_kl->Cdbo += (CEtors6 + CEconj3);

                /* dcos_theta_ijk */
                // rvec_ScaledAdd(workspace->f[i], CEtors7 + CEconj4, p_ijk_dcos_dk);
                // rvec_ScaledAdd(workspace->f[j], CEtors7 + CEconj4, p_ijk_dcos_dj);
                // rvec_ScaledAdd(workspace->f[k], CEtors7 + CEconj4, p_ijk_dcos_di);
                rvec_ScaledAdd(my_f[i], CEtors7 + CEconj4, p_ijk_dcos_dk);
                rvec_ScaledAdd(my_f[j], CEtors7 + CEconj4, p_ijk_dcos_dj);
                rvec_ScaledAdd(my_f[k], CEtors7 + CEconj4, p_ijk_dcos_di);

                /* dcos_theta_jkl */
                // rvec_ScaledAdd(workspace->f[j], CEtors8 + CEconj5, p_jkl_dcos_di);
                // rvec_ScaledAdd(workspace->f[k], CEtors8 + CEconj5, p_jkl_dcos_dj);
                // rvec_ScaledAdd(workspace->f[l], CEtors8 + CEconj5, p_jkl_dcos_dk);
                rvec_ScaledAdd(my_f[j], CEtors8 + CEconj5, p_jkl_dcos_di);
                rvec_ScaledAdd(my_f[k], CEtors8 + CEconj5, p_jkl_dcos_dj);
                rvec_ScaledAdd(my_f[l], CEtors8 + CEconj5, p_jkl_dcos_dk);
                /* dcos_omega */
                // rvec_ScaledAdd(workspace->f[i], CEtors9 + CEconj6, dcos_omega_di);
                // rvec_ScaledAdd(workspace->f[j], CEtors9 + CEconj6, dcos_omega_dj);
                // rvec_ScaledAdd(workspace->f[k], CEtors9 + CEconj6, dcos_omega_dk);
                // rvec_ScaledAdd(workspace->f[l], CEtors9 + CEconj6, dcos_omega_dl);
                rvec_ScaledAdd(my_f[i], CEtors9 + CEconj6, dcos_omega_di);
                rvec_ScaledAdd(my_f[j], CEtors9 + CEconj6, dcos_omega_dj);
                rvec_ScaledAdd(my_f[k], CEtors9 + CEconj6, dcos_omega_dk);
                rvec_ScaledAdd(my_f[l], CEtors9 + CEconj6, dcos_omega_dl);

                // tally
                // if (system->pair_ptr->eflag_either)
                // {
                //   eng_tmp = e_tor + e_con;
                //   // system->pair_ptr->eng_vdwl += eng_tmp;
                //   // eng_virial[5] += eng_tmp;
                //   total_e_vdwl += eng_tmp;
                // }

              } // if-fbh->cnt

            }//for-pl

          }//for-pi

        }//if-len-pk


    }//for-pk

  }//for-j
}// end parallel region   
  data->my_en.e_ang += total_e_ang; 
  data->my_en.e_pen += total_e_pen; 
  data->my_en.e_coa += total_e_coa; 
  data->my_en.e_tor += total_e_tor; 
  data->my_en.e_con += total_e_con;
  if (system->pair_ptr->eflag_either)
    system->pair_ptr->eng_vdwl += (total_e_ang + total_e_pen + total_e_coa + total_e_tor + total_e_con);
}
}//namespace
