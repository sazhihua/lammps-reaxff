// clang-format off
/*----------------------------------------------------------------------
  PuReMD - Purdue ReaxFF Molecular Dynamics Program

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

#include "reaxff_api.h"

#include <cmath>
#include <cstring>

#include "error.h"
#include "utils.h"

using LAMMPS_NS::utils::errorurl;

namespace ReaxFF {

  static void Compute_Bonded_Forces(reax_system *system,
                                    control_params *control,
                                    simulation_data *data,
                                    storage *workspace,
                                    reax_list **lists)
  {
    BO(system, workspace, lists);
    Bonds(system, data, workspace, lists);
    Atom_Energy(system, control, data, workspace, lists);
    Valence_and_Torsion_Angles(system, control, data, workspace, lists);
    if (control->hbond_cut > 0)
      Hydrogen_Bonds(system, data, workspace, lists);
  }

  static void Compute_NonBonded_Forces(reax_system *system,
                                       control_params *control,
                                       simulation_data *data,
                                       storage *workspace,
                                       reax_list **lists)
  {

    if (control->tabulate != 0)
      system->error_ptr->all(FLERR, "Scalar ablation list redesign requires tabulate 0");
    /* van der Waals and Coulomb interactions */
    if (control->tabulate == 0)
      vdW_Coulomb_Energy(system, control, data, workspace, lists);
    else
      Tabulated_vdW_Coulomb_Energy(system, control, data, workspace, lists);
  }

  static void Compute_Total_Force(reax_system *system, storage *workspace, reax_list **lists)
  {
    int i, pj;
    reax_list *bonds = (*lists) + BONDS;

    bond_data *nbr_j, *nbr_k;
    bond_order_data *bo_ij, *bo_ji;
    dbond_coefficients coef;
    int pk, k, j;
    /* Virial Tallying variables */
    rvec fi_tmp, fj_tmp, fk_tmp, delij, delji, delki, delkj, temp;

    double isum_c = 0;


    for (i = 0; i < system->N; ++i)
    {
      isum_c = 0;
      for (pj = Start_Index(i, bonds); pj < End_Index(i, bonds); ++pj)
      {
        //if (i < bonds->select.bond_list[pj].nbr) // <=> i < j;
        {
          //Add_dBond_to_Forces(system, i, pj, workspace, lists);

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

          //force on j: 
          //c = -(coef.C1dbo + coef.C1dDelta + coef.C2dbopi + coef.C2dbopi2);
          //rvec_Scale(    temp, c,    bo_ij->dBOp);

          //c = (coef.C3dbo + coef.C3dDelta + coef.C4dbopi + coef.C4dbopi2);
          //rvec_ScaledAdd(temp,  c,    workspace->dDeltap_self[j]);

          //rvec_ScaledAdd(temp, -coef.C1dbopi,  bo_ij->dln_BOp_pi);
          //rvec_ScaledAdd(temp, -coef.C1dbopi2, bo_ij->dln_BOp_pi2);

          //rvec_Add(workspace->f[j], temp);
         
          
          isum_c += -(coef.C2dbo + coef.C2dDelta + coef.C3dbopi + coef.C3dbopi2);

          //// forces on k: i neighbor
          //for (pk = Start_Index(i, bonds); pk < End_Index(i, bonds); ++pk) 
          //{
          //  nbr_k = &(bonds->select.bond_list[pk]);
          //  k = nbr_k->nbr;

          //  const double c = -(coef.C2dbo + coef.C2dDelta + coef.C3dbopi + coef.C3dbopi2);
          //  rvec_Scale(temp, c, nbr_k->bo_data.dBOp);
          //  rvec_Add(workspace->f[k], temp);
          //}


          //// forces on k: j neighbor
          //for (pk = Start_Index(j, bonds); pk < End_Index(j, bonds); ++pk) 
          //{
          //  nbr_k = &(bonds->select.bond_list[pk]);
          //  k = nbr_k->nbr;

          //  const double c = -(coef.C3dbo + coef.C3dDelta + coef.C4dbopi + coef.C4dbopi2);
          //  rvec_Scale(temp, c, nbr_k->bo_data.dBOp);
          //  rvec_Add(workspace->f[k], temp);
          //}

        
        }//if
        //else
        //  printf("i = %d, j = %d\n ", bonds->select.bond_list[pj].nbr);

      }//for-pj


      // forces on k: i neighbor
      for (pk = Start_Index(i, bonds); pk < End_Index(i, bonds); ++pk) 
      {
        nbr_k = &(bonds->select.bond_list[pk]);
        k = nbr_k->nbr;
        workspace->f[k][0] += isum_c * nbr_k->bo_data.dBOp[0];
        workspace->f[k][1] += isum_c * nbr_k->bo_data.dBOp[1];
        workspace->f[k][2] += isum_c * nbr_k->bo_data.dBOp[2];
      }



    }//for-i
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
    int btop_i, num_bonds, num_hbonds;
    int ihb, jhb, ihb_top, jhb_top;
    int local, flag;
    double cutoff;
    reax_list *far_nbrs, *bonds, *hbonds;
    single_body_parameters *sbp_i, *sbp_j;
    two_body_parameters *twbp;
    far_neighbor_data *nbr_pj;
    reax_atom *atom_i, *atom_j;

    far_nbrs = *lists + FAR_NBRS;
    bonds = *lists + BONDS;
    hbonds = *lists + HBONDS;

    for (i = 0; i < system->n; ++i)
      workspace->bond_mark[i] = 0;
    for (i = system->n; i < system->N; ++i) {
      workspace->bond_mark[i] = 1000; // put ghost atoms to an infinite distance
    }

    num_bonds = 0;
    num_hbonds = 0;
    btop_i = 0;
    
    cutoff = control->bond_cut;
    double bo_cut = control->bo_cut;

    for (i = 0; i < system->N; ++i) 
    {
      atom_i = &(system->my_atoms[i]);
      type_i  = atom_i->type;
      if (type_i < 0) continue;
      start_i = Start_Index(i, far_nbrs);
      end_i   = End_Index(i, far_nbrs);
      btop_i = End_Index(i, bonds);
      sbp_i = &(system->reax_param.sbp[type_i]);

      ihb = -1;
      ihb_top = -1;
      if (i < system->n && control->hbond_cut > 0) {
        ihb = sbp_i->p_hbond;
        if (ihb == 1)
          ihb_top = End_Index(atom_i->Hindex, hbonds);
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

        /* Cached hydrogen bonds use the same full neighbor traversal. */
        if (ihb == 1 && nbr_pj->d <= control->hbond_cut) {
          jhb = sbp_j->p_hbond;
          if (jhb == 2) {
            hbonds->select.hbond_list[ihb_top].nbr = j;
            hbonds->select.hbond_list[ihb_top].scl = 1;
            hbonds->select.hbond_list[ihb_top].ptr = nbr_pj;
            ++ihb_top;
            ++num_hbonds;
          }
        }
        twbp = &(system->reax_param.tbp[type_i][type_j]);

        


        /********** BOp() **********/

        int j, btop_j;
        double rr2, C12, C34, C56;
        double Cln_BOp_s, Cln_BOp_pi, Cln_BOp_pi2;
        double BO, BO_s, BO_pi, BO_pi2;
        bond_data *ibond, *jbond;
        bond_order_data *bo_ij, *bo_ji;

        j = nbr_pj->nbr;
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
        
        {












        }//BOp()

        if (nbr_pj->d <= control->bond_cut &&  BO >= bo_cut) 
        {
          /****** bonds i-j and j-i ******/
          ibond = &(bonds->select.bond_list[btop_i]);
          ibond->nbr = j;
          ibond->d = nbr_pj->d;
          rvec_Copy(ibond->dvec, nbr_pj->dvec);
          ivec_Copy(ibond->rel_box, nbr_pj->rel_box);
          ibond->dbond_index = btop_i;
          

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
      if (ihb == 1)
        Set_End_Index(atom_i->Hindex, ihb_top, hbonds);

    }//for-i


    workspace->realloc.num_bonds = num_bonds;

    workspace->realloc.num_hbonds = num_hbonds;
    Validate_Lists(system, lists, data->step, system->N, system->numH);

  }


  void Estimate_Storages(reax_system *system, control_params *control,
                          reax_list **lists, int *Htop, int *hb_top,
                          int *bond_top, int *num_3body)
  {
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

    for (i = 0; i < system->N; ++i) {
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
            if (control->hbond_cut > 0.1 && (ihb==1 || ihb==2) &&
                nbr_pj->d <= control->hbond_cut) {
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
      *num_3body += SQR(bond_top[i]);
      bond_top[i] = MAX(bond_top[i] * 2, MIN_BONDS);
    }

  }

  void Compute_Forces(reax_system *system, control_params *control,
                      simulation_data *data, storage *workspace,
                      reax_list **lists)
  {

    Init_Forces_noQEq(system, control, data, workspace, lists);

    /********* bonded interactions ************/
    Compute_Bonded_Forces(system, control, data, workspace, lists);

    /********* nonbonded interactions ************/
    Compute_NonBonded_Forces(system, control, data, workspace, lists);

    /*********** total force ***************/
    Compute_Total_Force(system, workspace, lists);
  }
}
