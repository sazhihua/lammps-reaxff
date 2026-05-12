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
#include <cstring>
#include <omp.h>

#include "error.h"
#include "utils.h"
// #include "gptl.h"
#include "reaxff_api_sve.h"

using namespace LAMMPS_NS;
using LAMMPS_NS::utils::errorurl;

namespace ReaxFFSVE {


int cmp_bonds(const void *a, const void *b)
{
  bond_data *abond = (bond_data *)a;
  bond_data *bbond = (bond_data *)b;
  if(abond->nbr < bbond->nbr)
    return -1;

  if(abond->nbr > bbond->nbr)
    return 1;

  return 0;
}


static void sort_bonds(reax_list *bonds, reax_system *system)
{
  int i, j;
  for(i = 0; i < system->N; i++)
  {
    int i_start = Start_Index(i, bonds);
    int i_stop = End_Index(i, bonds);
    qsort(bonds->select.bond_list + i_start, i_stop-i_start, sizeof(bond_data), cmp_bonds);
  }

  for(i = 0; i < system->N; i++)
  {
    int start_i = Start_Index(i, bonds);
    int end_i = End_Index(i, bonds);
    int pj;

    for(pj = start_i; pj < end_i; pj++)
    {
      bond_data *jbond = bonds->select.bond_list + pj;
      j = jbond->nbr;
      int start_j = Start_Index(j, bonds);
      int end_j = End_Index(j, bonds);
      int pk;
      int flag = 0;

      for(pk = start_j; pk < end_j; pk++)
      {
        bond_data *kbond = bonds->select.bond_list + pk;
        int k = kbond->nbr;
        if(k == i)
        {
          jbond->sym_index = pk;
          flag = 1;
          break;
        }

      }//for-pk
      if(flag == 0) printf("sym_index not found: i = %d, j = %d\n", i, j);

    }//fpr-pj

  }//for-i

}


  static void Compute_Bonded_Forces(reax_system *system,
                                    control_params *control,
                                    simulation_data *data,
                                    storage *workspace,
                                    reax_list **lists)
  {
    BO(system, workspace, lists);
    Bonds(system, data, workspace, lists);
    Atom_Energy(system, control, data, workspace, lists);

    //Valence_Angles(system, control, data, workspace, lists);
    //Torsion_Angles(system, control, data, workspace, lists);
    Valence_and_Torsion_Angles(system, control, data, workspace, lists);

    if (control->hbond_cut > 0)
      Hydrogen_Bonds_nohlist(system, data, workspace, lists, control);
      //Hydrogen_Bonds(system, data, workspace, lists);
    
  }

  static void Compute_NonBonded_Forces(reax_system *system,
                                       control_params *control,
                                       simulation_data *data,
                                       storage *workspace,
                                       reax_list **lists)
  {

    /* van der Waals and Coulomb interactions */
    //if (control->tabulate == 0)
      vdW_Coulomb_Energy(system, control, data, workspace, lists);
    //else
    //  Tabulated_vdW_Coulomb_Energy(system, control, data, workspace, lists);
  }

  static void Compute_Total_Force(reax_system *system, storage *workspace, reax_list **lists)
  {
    reax_list *bonds = (*lists) + BONDS;

    int nthr = omp_get_max_threads();

  #pragma omp parallel for num_threads(nthr) 
    for (int i = 0; i < system->N; ++i) {
      for (int j = 0; j < nthr; ++j)
        workspace->CdDelta[i] += workspace->CdDeltaTmp[j][i];
    }
    
  #pragma omp parallel num_threads(nthr)
  {
    int ithr = omp_get_thread_num();
    double (*my_f)[3] = (double(*)[3])workspace->forceTmp[ithr][0];

    #pragma omp for schedule(dynamic, 64)
    for (int i = 0; i < system->N; ++i)
    {
      if (workspace->bond_mark[i] > 3) {
        continue;
      }
      double isum_c = 0;
      for (int pj = Start_Index(i, bonds); pj < End_Index(i, bonds); ++pj) {
          /* Initializations */
          bond_data* nbr_j = &(bonds->select.bond_list[pj]);
          int j = nbr_j->nbr;
          bond_order_data *bo_ij = &(nbr_j->bo_data);
          bond_order_data *bo_ji = &(bonds->select.bond_list[nbr_j->sym_index].bo_data);
          // double bo_ji;
          double c = bo_ij->Cdbo + bo_ji->Cdbo;
          dbond_coefficients coef;
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
          /* Virial Tallying variables */
          rvec temp;
          rvec_Scale(    temp, c,    bo_ij->dBOp);

          c = (coef.C2dbo + coef.C2dDelta + coef.C3dbopi + coef.C3dbopi2);
          rvec_ScaledAdd(temp, c,    workspace->dDeltap_self[i]);

          rvec_ScaledAdd(temp, coef.C1dbopi,  bo_ij->dln_BOp_pi);
          rvec_ScaledAdd(temp, coef.C1dbopi2, bo_ij->dln_BOp_pi2);

          rvec_Add(my_f[i], temp);
          
          isum_c += -(coef.C2dbo + coef.C2dDelta + coef.C3dbopi + coef.C3dbopi2);

      }//for-pj


      // forces on k: i neighbor
      for (int pk = Start_Index(i, bonds); pk < End_Index(i, bonds); ++pk) 
      {
        bond_data *nbr_k = &(bonds->select.bond_list[pk]);
        int k = nbr_k->nbr;
        my_f[k][0] += isum_c * nbr_k->bo_data.dBOp[0];
        my_f[k][1] += isum_c * nbr_k->bo_data.dBOp[1];
        my_f[k][2] += isum_c * nbr_k->bo_data.dBOp[2];
      }

    }//for-i
  }

    #pragma omp parallel for num_threads(nthr)
    for (int i = 0; i < system->N; ++i) {
      for (int j = 0; j < nthr; ++j) {
        workspace->f[i][0] += workspace->forceTmp[j][i][0];
        workspace->f[i][1] += workspace->forceTmp[j][i][1];
        workspace->f[i][2] += workspace->forceTmp[j][i][2];
        workspace->forceTmp[j][i][0] = 0.0;
        workspace->forceTmp[j][i][1] = 0.0;
        workspace->forceTmp[j][i][2] = 0.0;
      }
    }

    #pragma omp parallel for num_threads(nthr)
    for (int tid = 0; tid < nthr; tid++) {
      memset(workspace->CdDeltaTmp[tid], 0.0, sizeof(double) * system->N);
      // memset(workspace->forceTmp[tid][0], 0.0, sizeof(double) * system->N * 3);
    }

  }

  static void Validate_Lists(reax_system *system, reax_list **lists,
                             int step, int N, int numH)
  {
    int i, comp, Hindex;
    reax_list *bonds, *hbonds;

    double saferzone = system->saferzone;

    /* bond list */
    if (N > 0) {
      bonds = *lists + BONDS;

      for (i = 0; i < N; ++i) {
        system->my_atoms[i].num_bonds = MAX(Num_Entries(i,bonds)*2, MIN_BONDS);

        if (i < N-1)
          comp = Start_Index(i+1, bonds);
        else comp = bonds->num_intrs;

        if (End_Index(i, bonds) > comp)
          system->error_ptr->one(FLERR, fmt::format("step {}: bondchk failed: i={} end(i)={} "
                                                    "str(i+1)={}{}", step, i ,End_Index(i,bonds),
                                                    comp, errorurl(18)));
      }
    }


    /* hbonds list */
    if (numH > 0) {
      hbonds = *lists + HBONDS;

      for (i = 0; i < N; ++i) {
        Hindex = system->my_atoms[i].Hindex;
        if (Hindex > -1) {
          system->my_atoms[i].num_hbonds =
            (int)(MAX(Num_Entries(Hindex, hbonds)*saferzone, system->minhbonds));

          if (Hindex < numH-1)
            comp = Start_Index(Hindex+1, hbonds);
          else comp = hbonds->num_intrs;

          if (End_Index(Hindex, hbonds) > comp)
            system->error_ptr->one(FLERR, fmt::format("step {}: hbondchk failed: H={} end(H)={} "
                                                      "str(H+1)={}{}", step, Hindex,
                                                      End_Index(Hindex,hbonds),comp,errorurl(18)));
        }
      }
    }
  }

  static void Init_Forces_noQEq(reax_system *system, control_params *control,
                                simulation_data *data, storage *workspace,
                                reax_list **lists) 
 {
    int i, j, pj;
    int start_i, end_i;
    int type_i, type_j;
    int btop_i, num_bonds = 0;
    int local, flag;
    double cutoff;
    reax_list *far_nbrs, *bonds;
    single_body_parameters *sbp_i, *sbp_j;
    two_body_parameters *twbp;
    far_neighbor_data *nbr_pj;
    reax_atom *atom_i, *atom_j;
    
    far_nbrs = *lists + FAR_NBRS;
    bonds = *lists + BONDS;
    num_bonds = 0;
    btop_i = 0;
    
    cutoff = control->bond_cut;
    if(cutoff <= 0) return;
    double bo_cut = control->bo_cut;
    
    btop_i = 0;
    for (i = 0; i < system->N; ++i) 
    {
      if (workspace->bond_mark[i] > 3) continue;
      atom_i = &(system->my_atoms[i]);
      type_i  = atom_i->type;
      if (type_i < 0) continue;
      start_i = Start_Index(i, far_nbrs);
      end_i   = End_Index(i, far_nbrs);
      Set_Start_Index(i, btop_i, bonds);
      sbp_i = &(system->reax_param.sbp[type_i]);
        
      /* update i-j distance - check if j is within cutoff */
      for (pj = start_i; pj < end_i; ++pj) 
      {
        nbr_pj = &(far_nbrs->select.far_nbr_list[pj]);
        j = nbr_pj->nbr;
        if (workspace->bond_mark[j] > 3) continue;
        
        atom_j = &(system->my_atoms[j]);

        type_j = atom_j->type;
        if (type_j < 0) continue;
        sbp_j = &(system->reax_param.sbp[type_j]);
        twbp = &(system->reax_param.tbp[type_i][type_j]);
            
        /********** BOp() **********/

        int j, btop_j;
        double rr2, C12, C34, C56;
        double Cln_BOp_s, Cln_BOp_pi, Cln_BOp_pi2;
        double BO, BO_s, BO_pi, BO_pi2;
        bond_data *ibond, *jbond;
        bond_order_data *bo_ij, *bo_ji;
            
        j = nbr_pj->nbr;
        // if (nbr_pj->d > twbp->r_bocut || tmp[j] > 3) continue;
        rr2 = 1.0 / SQR(nbr_pj->d);
        
        if (sbp_i->r_s > 0.0 && sbp_j->r_s > 0.0) {
          C12 = twbp->p_bo1 * pow(nbr_pj->d / twbp->r_s, twbp->p_bo2);
          BO_s = (1.0 + bo_cut) * exp(C12);
        } else BO_s = C12 = 0.0;
        
        if (sbp_i->r_pi > 0.0 && sbp_j->r_pi > 0.0) {
          C34 = twbp->p_bo3 * pow(nbr_pj->d / twbp->r_p, twbp->p_bo4);
          BO_pi = exp(C34);
        } else BO_pi = C34 = 0.0;
        
        if (sbp_i->r_pi_pi > 0.0 && sbp_j->r_pi_pi > 0.0) {
          C56 = twbp->p_bo5 * pow(nbr_pj->d / twbp->r_pp, twbp->p_bo6);
          BO_pi2= exp(C56);
        } else BO_pi2 = C56 = 0.0;
            
        /* Initially BO values are the uncorrected ones, page 1 */
        BO = BO_s + BO_pi + BO_pi2;

        ///****************** BOP_single() *****************/
        if (nbr_pj->d <= control->bond_cut &&  BO >= bo_cut) 
        {
          /****** bonds i-j and j-i ******/
          ibond = &(bonds->select.bond_list[btop_i]);
          ibond->nbr = j;
          ibond->d = nbr_pj->d;
          rvec_Copy(ibond->dvec, nbr_pj->dvec);
          ivec_Copy(ibond->rel_box, nbr_pj->rel_box);
          ibond->dbond_index = btop_i;
          //ibond->sym_index = btop_j;
          
          //printf("i = %d, j = %d, btop_j = %d\n", i, j, btop_j);
                
          bo_ij = &(ibond->bo_data);
          bo_ij->BO = BO;
          bo_ij->BO_s = BO_s;
          bo_ij->BO_pi = BO_pi;
          bo_ij->BO_pi2 = BO_pi2;
                
          /* Bond Order page2-3, derivative of total bond order prime */
          Cln_BOp_s = twbp->p_bo2 * C12 * rr2;
          Cln_BOp_pi = twbp->p_bo4 * C34 * rr2;
          Cln_BOp_pi2 = twbp->p_bo6 * C56 * rr2;

          /* Only dln_BOp_xx wrt. dr_i is stored here, note that
             dln_BOp_xx/dr_i = -dln_BOp_xx/dr_j and all others are 0 */
          rvec_Scale(bo_ij->dln_BOp_s,-bo_ij->BO_s*Cln_BOp_s,ibond->dvec);
          rvec_Scale(bo_ij->dln_BOp_pi,-bo_ij->BO_pi*Cln_BOp_pi,ibond->dvec);
          rvec_Scale(bo_ij->dln_BOp_pi2, -bo_ij->BO_pi2*Cln_BOp_pi2,ibond->dvec);

          rvec_Scale(bo_ij->dBOp,
                      -(bo_ij->BO_s * Cln_BOp_s +
                        bo_ij->BO_pi * Cln_BOp_pi +
                        bo_ij->BO_pi2 * Cln_BOp_pi2), ibond->dvec);
                
          rvec_Add(workspace->dDeltap_self[i], bo_ij->dBOp);

          bo_ij->BO_s -= bo_cut;
          bo_ij->BO -= bo_cut;
          workspace->total_bond_order[i] += bo_ij->BO; //currently total_BOp
          bo_ij->Cdbo = bo_ij->Cdbopi = bo_ij->Cdbopi2 = 0.0;
                
          num_bonds += 1;
          ++btop_i;

        }//BOp_single()


      }//for-pj

      Set_End_Index(i, btop_i, bonds);

    }//for-i

    workspace->realloc.num_bonds = num_bonds;
  }

static void Init_Forces_noQEq_omp(reax_system *system, control_params *control,
                             simulation_data *data, storage *workspace,
                             reax_list **lists)
{
  int i, j, pj;
  int start_i, end_i;
  int type_i, type_j;
  double cutoff = control->bond_cut;
  double bo_cut = control->bo_cut;

  reax_list *far_nbrs = *lists + FAR_NBRS;
  reax_list *bonds    = *lists + BONDS;

  int N = system->N;

  /* ---------------- Pass 1: count bonds per atom ---------------- */

  // int *bond_count = (int *) malloc(sizeof(int) * N);
  int *bond_count = workspace->bond_count;

#pragma omp parallel for schedule(dynamic, 64)
  for (i = 0; i < N; ++i) {
    int cnt = 0;

    if (workspace->bond_mark[i] > 3) {
      bond_count[i] = 0;
      continue;
    }

    reax_atom *atom_i = &(system->my_atoms[i]);
    type_i = atom_i->type;
    if (type_i < 0) {
      bond_count[i] = 0;
      continue;
    }

    single_body_parameters *sbp_i =
        &(system->reax_param.sbp[type_i]);

    start_i = Start_Index(i, far_nbrs);
    end_i   = End_Index(i, far_nbrs);

    for (pj = start_i; pj < end_i; ++pj) {
      far_neighbor_data *nbr_pj =
          &(far_nbrs->select.far_nbr_list[pj]);

      j = nbr_pj->nbr;
      if (workspace->bond_mark[j] > 3) continue;

      reax_atom *atom_j = &(system->my_atoms[j]);
      type_j = atom_j->type;
      if (type_j < 0) continue;

      single_body_parameters *sbp_j =
          &(system->reax_param.sbp[type_j]);
      two_body_parameters *twbp =
          &(system->reax_param.tbp[type_i][type_j]);

      /* ---- compute uncorrected bond order ---- */

      double C12 = 0.0, C34 = 0.0, C56 = 0.0;
      double BO_s = 0.0, BO_pi = 0.0, BO_pi2 = 0.0;

      if (sbp_i->r_s > 0.0 && sbp_j->r_s > 0.0) {
        C12 = twbp->p_bo1 *
              pow(nbr_pj->d / twbp->r_s, twbp->p_bo2);
        BO_s = (1.0 + bo_cut) * exp(C12);
      }

      if (sbp_i->r_pi > 0.0 && sbp_j->r_pi > 0.0) {
        C34 = twbp->p_bo3 *
              pow(nbr_pj->d / twbp->r_p, twbp->p_bo4);
        BO_pi = exp(C34);
      }

      if (sbp_i->r_pi_pi > 0.0 && sbp_j->r_pi_pi > 0.0) {
        C56 = twbp->p_bo5 *
              pow(nbr_pj->d / twbp->r_pp, twbp->p_bo6);
        BO_pi2 = exp(C56);
      }

      double BO = BO_s + BO_pi + BO_pi2;

      if (nbr_pj->d <= cutoff && BO >= bo_cut)
        cnt++;
    }

    bond_count[i] = cnt;
  }

  /* ---------------- Prefix sum ---------------- */

  // int *bond_start = (int *) malloc(sizeof(int) * N);
  int *bond_start = workspace->bond_start;
  bond_start[0] = 0;
  for (i = 1; i < N; ++i)
    bond_start[i] = bond_start[i - 1] + bond_count[i - 1];

  int num_bonds = bond_start[N - 1] + bond_count[N - 1];
  workspace->realloc.num_bonds = num_bonds;

  /* ---------------- Pass 2: write bonds ---------------- */

#pragma omp parallel for schedule(dynamic, 64)
  for (i = 0; i < N; ++i) {
    if (bond_count[i] == 0) {
      Set_Start_Index(i, bond_start[i], bonds);
      Set_End_Index(i, bond_start[i], bonds);
      continue;
    }

    if (workspace->bond_mark[i] > 3) {
      Set_Start_Index(i, bond_start[i], bonds);
      Set_End_Index(i, bond_start[i], bonds);
      continue;
    }

    reax_atom *atom_i = &(system->my_atoms[i]);
    type_i = atom_i->type;
    if (type_i < 0) {
      Set_Start_Index(i, bond_start[i], bonds);
      Set_End_Index(i, bond_start[i], bonds);
      continue;
    }

    single_body_parameters *sbp_i =
        &(system->reax_param.sbp[type_i]);

    int btop_i = bond_start[i];
    Set_Start_Index(i, btop_i, bonds);

    start_i = Start_Index(i, far_nbrs);
    end_i   = End_Index(i, far_nbrs);

    for (pj = start_i; pj < end_i; ++pj) {
      far_neighbor_data *nbr_pj =
          &(far_nbrs->select.far_nbr_list[pj]);

      j = nbr_pj->nbr;
      if (workspace->bond_mark[j] > 3) continue;

      reax_atom *atom_j = &(system->my_atoms[j]);
      type_j = atom_j->type;
      if (type_j < 0) continue;

      single_body_parameters *sbp_j =
          &(system->reax_param.sbp[type_j]);
      two_body_parameters *twbp =
          &(system->reax_param.tbp[type_i][type_j]);

      double rr2 = 1.0 / (nbr_pj->d * nbr_pj->d);

      double C12 = 0.0, C34 = 0.0, C56 = 0.0;
      double BO_s = 0.0, BO_pi = 0.0, BO_pi2 = 0.0;

      if (sbp_i->r_s > 0.0 && sbp_j->r_s > 0.0) {
        C12 = twbp->p_bo1 *
              pow(nbr_pj->d / twbp->r_s, twbp->p_bo2);
        BO_s = (1.0 + bo_cut) * exp(C12);
      }

      if (sbp_i->r_pi > 0.0 && sbp_j->r_pi > 0.0) {
        C34 = twbp->p_bo3 *
              pow(nbr_pj->d / twbp->r_p, twbp->p_bo4);
        BO_pi = exp(C34);
      }

      if (sbp_i->r_pi_pi > 0.0 && sbp_j->r_pi_pi > 0.0) {
        C56 = twbp->p_bo5 *
              pow(nbr_pj->d / twbp->r_pp, twbp->p_bo6);
        BO_pi2 = exp(C56);
      }

      double BO = BO_s + BO_pi + BO_pi2;

      if (nbr_pj->d <= cutoff && BO >= bo_cut) {
        bond_data *ibond =
            &(bonds->select.bond_list[btop_i]);

        ibond->nbr = j;
        ibond->d   = nbr_pj->d;
        rvec_Copy(ibond->dvec, nbr_pj->dvec);
        ivec_Copy(ibond->rel_box, nbr_pj->rel_box);
        ibond->dbond_index = btop_i;

        bond_order_data *bo_ij = &(ibond->bo_data);

        bo_ij->BO     = BO;
        bo_ij->BO_s   = BO_s;
        bo_ij->BO_pi  = BO_pi;
        bo_ij->BO_pi2 = BO_pi2;

        double Cln_BOp_s   = twbp->p_bo2 * C12 * rr2;
        double Cln_BOp_pi  = twbp->p_bo4 * C34 * rr2;
        double Cln_BOp_pi2 = twbp->p_bo6 * C56 * rr2;

        rvec_Scale(bo_ij->dln_BOp_s,
                   -bo_ij->BO_s * Cln_BOp_s, ibond->dvec);
        rvec_Scale(bo_ij->dln_BOp_pi,
                   -bo_ij->BO_pi * Cln_BOp_pi, ibond->dvec);
        rvec_Scale(bo_ij->dln_BOp_pi2,
                   -bo_ij->BO_pi2 * Cln_BOp_pi2, ibond->dvec);

        rvec_Scale(bo_ij->dBOp,
                   -(bo_ij->BO_s * Cln_BOp_s +
                     bo_ij->BO_pi * Cln_BOp_pi +
                     bo_ij->BO_pi2 * Cln_BOp_pi2),
                   ibond->dvec);

        rvec_Add(workspace->dDeltap_self[i], bo_ij->dBOp);

        bo_ij->BO_s -= bo_cut;
        bo_ij->BO   -= bo_cut;

        workspace->total_bond_order[i] += bo_ij->BO;
        bo_ij->Cdbo = bo_ij->Cdbopi = bo_ij->Cdbopi2 = 0.0;

        ++btop_i;
      }
    }

    Set_End_Index(i, btop_i, bonds);
  }

}



static void Init_Forces_noQEq_Hbond(reax_system *system, control_params *control,
                                simulation_data *data, storage *workspace,
                                reax_list **lists) 
 {
    int i, j, pj;
    int start_i, end_i;
    int type_i, type_j;
    int /* btop_i, num_bonds,*/ num_hbonds;
    int ihb, jhb, ihb_top, jhb_top;
    double cutoff;
    reax_list *far_nbrs,/* *bonds,*/ *hbonds;
    single_body_parameters *sbp_i, *sbp_j;
    far_neighbor_data *nbr_pj;
    reax_atom *atom_i, *atom_j;

    far_nbrs = *lists + FAR_NBRS;
    hbonds = *lists + HBONDS;

    num_hbonds = 0;
    cutoff = control->hbond_cut;
    if(cutoff <= 0 ) return;

    for (i = 0; i < system->n; ++i) //
    {
      atom_i = &(system->my_atoms[i]);
      type_i  = atom_i->type;
      if (type_i < 0) continue;
      start_i = Start_Index(i, far_nbrs);
      end_i   = End_Index(i, far_nbrs);
      sbp_i = &(system->reax_param.sbp[type_i]);

      
      ihb = -1;
      ihb_top = -1;
      ihb = sbp_i->p_hbond;
      if (ihb == 1)
        ihb_top = End_Index(atom_i->Hindex, hbonds);
      else 
      {
        ihb_top = -1;
        continue;
      }

      /* update i-j distance - check if j is within cutoff */
      for (pj = start_i; pj < end_i; ++pj) 
      {
        nbr_pj = &(far_nbrs->select.far_nbr_list[pj]);
        j = nbr_pj->nbr;
        atom_j = &(system->my_atoms[j]);

        type_j = atom_j->type;
        if (type_j < 0) continue;
        sbp_j = &(system->reax_param.sbp[type_j]);

        /* hydrogen bond lists */
        if ((ihb==1 || ihb==2) && nbr_pj->d <= control->hbond_cut) 
        {
          jhb = sbp_j->p_hbond;
          if (ihb == 1 && jhb == 2) 
          {
            hbonds->select.hbond_list[ihb_top].nbr = j;
            hbonds->select.hbond_list[ihb_top].scl = 1;
            hbonds->select.hbond_list[ihb_top].ptr = nbr_pj;
            ++ihb_top;
            ++num_hbonds;
          }
          //else if (j < system->n && ihb == 2 && jhb == 1) 
          //{
          //  jhb_top = End_Index(atom_j->Hindex, hbonds);
          //  hbonds->select.hbond_list[jhb_top].nbr = i;
          //  hbonds->select.hbond_list[jhb_top].scl = -1;
          //  hbonds->select.hbond_list[jhb_top].ptr = nbr_pj;
          //  Set_End_Index(atom_j->Hindex, jhb_top+1, hbonds);
          //  ++num_hbonds;
          //}
        }

      }//for-pj

      if (ihb == 1)
        Set_End_Index(atom_i->Hindex, ihb_top, hbonds);
    }//for-i


    workspace->realloc.num_hbonds = num_hbonds;
    //printf("num_hbonds = %d -------------\n", num_hbonds);

    Validate_Lists(system, lists, data->step, system->N, system->numH);
    
  }

  void Estimate_Storages(reax_system *system, control_params *control,
                          reax_list **lists, int *Htop, int *hb_top,
                          int *bond_top, int *num_3body)
  {
    // this part is no need to be multithreaded
    // GPTLstart("estimate storages");
    int i, j, pj;
    int start_i, end_i;
    int type_i, type_j;
    int ihb, jhb;
    int local;
    double cutoff;
    double r_ij;
    double C12, C34, C56;
    double BO, BO_s, BO_pi, BO_pi2;
    reax_list *far_nbrs;
    single_body_parameters *sbp_i, *sbp_j;
    two_body_parameters *twbp;
    far_neighbor_data *nbr_pj;
    reax_atom *atom_i, *atom_j;

    int mincap = system->mincap;
    double safezone = system->safezone;
    double saferzone = system->saferzone;

    far_nbrs = *lists + FAR_NBRS;
    *Htop = 0;
    memset(hb_top, 0, sizeof(int) * system->local_cap);
    memset(bond_top, 0, sizeof(int) * system->total_cap);
    *num_3body = 0;

    for (i = 0; i < system->N; ++i) 
    {
      atom_i = &(system->my_atoms[i]);
      type_i  = atom_i->type;
      if (type_i < 0) continue;
      start_i = Start_Index(i, far_nbrs);
      end_i   = End_Index(i, far_nbrs);
      sbp_i = &(system->reax_param.sbp[type_i]);

      if (i < system->n) {
        local = 1;
        cutoff = control->nonb_cut;
        ++(*Htop);
        ihb = sbp_i->p_hbond;
      } else {
        local = 0;
        cutoff = control->bond_cut;
        ihb = -1;
      }

      for (pj = start_i; pj < end_i; ++pj) {
        nbr_pj = &(far_nbrs->select.far_nbr_list[pj]);
        j = nbr_pj->nbr;
        atom_j = &(system->my_atoms[j]);

        if (nbr_pj->d <= cutoff) {
          type_j = system->my_atoms[j].type;
          if (type_j < 0) continue;
          r_ij = nbr_pj->d;
          sbp_j = &(system->reax_param.sbp[type_j]);
          twbp = &(system->reax_param.tbp[type_i][type_j]);

          if (local) {
            if (j < system->n || atom_i->orig_id < atom_j->orig_id) //tryQEq ||1
            ++(*Htop);

            /* hydrogen bond lists */
            if (control->hbond_cut > 0.1 && (ihb==1 || ihb==2) && nbr_pj->d <= control->hbond_cut) {
              jhb = sbp_j->p_hbond;
              if (ihb == 1 && jhb == 2)
                ++hb_top[i];
              else if (j < system->n && ihb == 2 && jhb == 1)
                ++hb_top[j];
              }
          }

          /* uncorrected bond orders */
          if (nbr_pj->d <= control->bond_cut) {
            if (sbp_i->r_s > 0.0 && sbp_j->r_s > 0.0) {
              C12 = twbp->p_bo1 * pow(r_ij / twbp->r_s, twbp->p_bo2);
              BO_s = (1.0 + control->bo_cut) * exp(C12);
            }
            else BO_s = C12 = 0.0;

            if (sbp_i->r_pi > 0.0 && sbp_j->r_pi > 0.0) {
              C34 = twbp->p_bo3 * pow(r_ij / twbp->r_p, twbp->p_bo4);
              BO_pi = exp(C34);
            }
            else BO_pi = C34 = 0.0;

            if (sbp_i->r_pi_pi > 0.0 && sbp_j->r_pi_pi > 0.0) {
              C56 = twbp->p_bo5 * pow(r_ij / twbp->r_pp, twbp->p_bo6);
              BO_pi2= exp(C56);
            }
            else BO_pi2 = C56 = 0.0;

            /* Initially BO values are the uncorrected ones, page 1 */
            BO = BO_s + BO_pi + BO_pi2;

            if (BO >= control->bo_cut) {
              ++bond_top[i];
              ++bond_top[j];
            }
          }
        }
      }
    }

    *Htop = (int)(MAX(*Htop * safezone, mincap * MIN_HENTRIES));
    for (i = 0; i < system->n; ++i)
      hb_top[i] = (int)(MAX(hb_top[i] * saferzone, system->minhbonds));

    for (i = 0; i < system->N; ++i) {
      bond_top[i] = MAX(bond_top[i] * 2, MIN_BONDS);
    }
    // GPTLstop("estimate storages");
  }

  void Compute_Forces(reax_system *system, control_params *control,
                      simulation_data *data, storage *workspace,
                      reax_list **lists)
  {

    // GPTLstart("Init_Forces_noQEq");
    if (omp_get_max_threads() == 1) {
      Init_Forces_noQEq(system, control, data, workspace, lists);
    } else {
      Init_Forces_noQEq_omp(system, control, data, workspace, lists);
    }
    // GPTLstop("Init_Forces_noQEq");
    //Init_Forces_noQEq_Hbond(system, control, data, workspace, lists);

    //sort_bonds(*lists + BONDS, system);


    /********* bonded interactions ************/
    // GPTLstart("Compute_Bonded_Forces");
    Compute_Bonded_Forces(system, control, data, workspace, lists);
    // GPTLstop("Compute_Bonded_Forces");

    /********* nonbonded interactions ************/
    // GPTLstart("Compute_NonBonded_Forces");
    Compute_NonBonded_Forces(system, control, data, workspace, lists);
    // GPTLstop("Compute_NonBonded_Forces");

    /*********** total force ***************/
    // GPTLstart("Compute_Total_Force");
    Compute_Total_Force(system, workspace, lists);
    // GPTLstop("Compute_Total_Force");
  }
}

