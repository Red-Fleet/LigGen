from vina import Vina
from openbabel import pybel as pb
from openbabel import openbabel as ob
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Draw
import random
import math
import re
import os
import numpy as np
from rdkit.Chem import RDConfig
import sys
sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
import sascorer
import rdkit.Chem
import utils as utils
from rdkit import RDLogger
import uuid

class SimulatedAnnealing:
    def __init__(self, fragments: list[str], vina: Vina):
        self.fragments = fragments
        self.vina:Vina = vina
        self.grid_box = None  #[0=> center, 1=>size]
        self.total_frag_screened = 0
        self.total_frag_rejected_mpc = 0
        self.obconversion = ob.OBConversion()
        self.temp_folder_path = 'temp'
        self.obconv = ob.OBConversion()

        if os.path.exists(self.temp_folder_path) == False or os.path.isdir(self.temp_folder_path)==False:
            os.mkdir(self.temp_folder_path) 

    def setTarget(self, target_pdbqt_path: str, grid_param: tuple[int, list[int, int, int], list[int, int, int]] = None):
        '''
        grid_param = (spacing, center, box)
        '''
        self.vina.set_receptor(target_pdbqt_path)

        if grid_param is not None:
            self.setGridMap(grid_param)

    def setGridMap(self, grid_param: tuple[int, list[int, int, int], list[int, int, int]]):
        '''
        grid_param = (spacing, center, box)
        '''
        self.grid_box = [grid_param[1], grid_param[2]]
        if grid_param[0] == None:
            self.vina.compute_vina_maps(center=grid_param[1], box_size=grid_param[2])
        else:
            self.vina.compute_vina_maps(center=grid_param[1], box_size=grid_param[2], spacing=grid_param[0])

    def generateRdkitConformer(self, mol:rdkit.Chem.rdchem.Mol) -> rdkit.Chem.rdchem.Mol:
        '''input is rdkit mol and conformer is added to to same input mol and mol is also returned
        if it fails it will return None
        '''
        mol = AllChem.AddHs(mol)
        sucess = AllChem.EmbedMolecule(mol)

        if sucess == -1:
            return None

        return mol

    def placeRdkitMolAtNewPoint(self, mol: rdkit.Chem.rdchem.Mol, atom_idx: int, new_x: float, new_y: float, new_z: float) ->  rdkit.Chem.rdchem.Mol:
        ''' For shifing Rdkit molecule(conformer), conformer should be created for this method to work
        '''
        c = mol.GetConformer()
        atom_point = c.GetAtomPosition(atom_idx)
        x_offset = new_x - atom_point.x
        y_offset = new_y - atom_point.y
        z_offset = new_z - atom_point.z

        mol = self.shiftRdkitMol(mol, x_offset, y_offset, z_offset)

        return mol


    def shiftRdkitMol(self, mol: rdkit.Chem.rdchem.Mol, x_offset: float, y_offset: float, z_offset: float)->  rdkit.Chem.rdchem.Mol:
        ''' For shifing Rdkit molecule, offeset is, conformer should be created for this method to work
        '''

        c = mol.GetConformer()

        # shift all atoms
        for i in range(c.GetNumAtoms()):
            pos = c.GetAtomPosition(i)
            pos.x += x_offset
            pos.y += y_offset
            pos.z += z_offset

            c.SetAtomPosition(i, pos)

        return mol

    def getSmilesIdx(self, smile):
        '''returns start and end indices of smile atoms'''
        result = []
        i = 0
        while i < len(smile): 
            # check length 2 atom
            if len(smile)>i+1 and smile[i:i+2] in utils.smiles_atoms: 
                start = i
                i += 1  
                # checking if digit is present after atom
                while len(smile) > i+1 and smile[i+1].isdigit():
                    i = i+1
                result.append((start, i))
            # check length 1 atom
            if smile[i] in utils.smiles_atoms:  
                start = i
                # checking if digit is present after atom
                while len(smile) > i+1 and smile[i+1].isdigit():
                    i = i+1
                result.append((start, i))
            
            i += 1

        return result



    def addFragmentRandomlyToLigandSmiles(self, lig_smile, frag_smile, end_prob):
        '''Return new_smile, atom Map and position/index of lig_smile at which fragment is added 
        '''
        # total atoms in ligand smiles
        avaliable_idxs = [(-1, -1)] # (-1,-1) so that at 0th place string can be added
        
        # for chain
        avaliable_idxs += self.getSmilesIdx(lig_smile)

        avaliable_idxs.append((len(lig_smile), len(lig_smile)-1)) # for end 

        # probablities
        chain_prob = 1-end_prob
        avaliable_idx_probs = []
        for i in range(len(avaliable_idxs)):
            if i == 0 or i==len(avaliable_idxs)-1: avaliable_idx_probs.append(end_prob/2)
            else: avaliable_idx_probs.append(chain_prob/(len(avaliable_idxs)-2))

        # random position
        position = random.choices(
            population=avaliable_idxs,
            weights=avaliable_idx_probs,
            k=1
        )[0]

        new_lig_smile = ''
        insert_len = len(frag_smile)
        if position == avaliable_idxs[0]:
            new_lig_smile = frag_smile + lig_smile
        elif position == avaliable_idxs[-1]:
            new_lig_smile =  lig_smile + frag_smile
        else:
            insert_len += 2
            new_lig_smile = lig_smile[0:position[1]+1] + '('+ frag_smile+')' + lig_smile[position[1]+1:]

        # creating map
        if position == avaliable_idxs[0]:
            start_lig = []

        start_lig = self.getSmilesIdx(new_lig_smile[:position[1]+1])
        mid_lig = self.getSmilesIdx(new_lig_smile[position[1]+1:position[1]+1+insert_len])
        end_lig = self.getSmilesIdx(new_lig_smile[position[1]+1+insert_len+1:])

        atomMap = []
        for i in range(len(start_lig)):
            atomMap.append((i, i))

        for i in range(len(end_lig)):
            atomMap.append((len(start_lig)+len(mid_lig)+i, len(start_lig)+i))
        

        return new_lig_smile, atomMap, position[1]


    def rdkitToPdbqt(self, mol: rdkit.Chem.rdchem.Mol)->str:
        '''return pdbqt string of given rdkit mol
        '''
        mol_block = Chem.MolToMolBlock(mol) # sdf format

        pymol = pb.readstring(format="sdf", string=mol_block)
        pymol.addh()
        _ = pymol.calccharges("gasteiger")
        pdbqt = pymol.write(format="pdbqt", opt={'r':self.obconv.OUTOPTIONS})
        
        # making root so that rigid ligand can be used by vina
        pdbqt = [l for l in pdbqt.split("\n") if l.startswith("TER")==False]
        pdbqt = ["ROOT"] + pdbqt + ["ENDROOT", "TORSDOF 0"]
        pdbqt = "\n".join(pdbqt)
        
        return pdbqt

    def getRandomFragment(self) -> str:
        idx = random.randint(0, len(self.fragments)-1)
        self.idx = idx
        return self.fragments[idx]
        
    
    def isLigInGridbox(self,  mol: rdkit.Chem.rdchem.Mol):
        '''
        Returns true if all atoms of ligands(mol) are present inside the grid box
        '''
        c = mol.GetConformer()
        center = self.grid_box[0]
        size = self.grid_box[1]
        
        x_min = center[0] - size[0] / 2
        x_max = center[0] + size[0] / 2
        y_min = center[1] - size[1] / 2
        y_max = center[1] + size[1] / 2
        z_min = center[2] - size[2] / 2
        z_max = center[2] + size[2] / 2

        for i in range(c.GetNumAtoms()):
            pos = c.GetAtomPosition(i)
            if ((x_min <= pos.x <= x_max) and (y_min <= pos.y <= y_max) and (z_min <= pos.z <= z_max)) == False:
                return False
        
        return True

    def getSAScore(self, mol: rdkit.Chem.rdchem.Mol):
        '''returns synthesizability score of given ligand'''
        return sascorer.calculateScore(mol)

    def updateMolCoordsFromPdbqt(self, mol, pdbqt_path):
        # method will read atom coordinates from and bold info from mol  and return new mol without hydrogen
        # will return mol on sucess , and None on failure

        mol = Chem.RemoveHs(mol) # removing h and returning new one

        with open(pdbqt_path) as f:
            pdbqt_atoms = []
            for l in f:
                l = l.replace("\n", "")
                if l != "" and l.startswith("ATOM"):
                    atom = utils.readDetailsFromPdbLine(l)
                    if atom['atom_name'] != 'H':
                        pdbqt_atoms.append(atom)


        if mol.GetNumAtoms() != len(pdbqt_atoms):
            print("Could not update coords: wrong number of atoms")
            return None
            #raise Exception("Could not update coords: wrong number of atoms", mol.GetNumAtoms(), len(pdbqt_atoms))
            
        c = mol.GetConformer()
            
        for i in range(len(pdbqt_atoms)):
            for i in range(c.GetNumAtoms()):
                if mol.GetAtomWithIdx(i).GetSymbol().lower() != pdbqt_atoms[i]['atom_name'].lower(): 

                    print("Could not update coords: wrong type of atom", mol.GetAtomWithIdx(i).GetSymbol(), pdbqt_atoms[i]['atom_name'])
                    #raise Exception("Could not update coords: wrong type of atom", mol.GetAtomWithIdx(i).GetSymbol(), pdbqt_atoms[i]['atom_name'])
                    return None
                
                pos = c.GetAtomPosition(i)
                pos.x = pdbqt_atoms[i]['x_coordinate']
                pos.y = pdbqt_atoms[i]['y_coordinate']
                pos.z = pdbqt_atoms[i]['z_coordinate']

                c.SetAtomPosition(i, pos)
        

        return mol

        
    
    def _simulatedAnnealing(self, 
                            old_ligand: str,
                            old_score: float, 
                            old_ligand_3d, 
                            initial_position: list, 
                            max_mw: float, 
                            temp: float, 
                            iter: int, 
                            alpha,
                            end_prob:float,
                            max_iter_at_state:int,
                            vina_score_weight:float,
                            details:list=None)-> tuple:
        if details is None: details = []
        mpc_reject = 0

        for i in range(int(max_iter_at_state)):

            frag = self.getRandomFragment()
            self.total_frag_screened += 1

            frag = utils.justifyRingCloserLabelInSmiles(in_smiles=frag, reference_smiles=old_ligand)
            new_lig, atomMap, frag_index = self.addFragmentRandomlyToLigandSmiles(lig_smile=old_ligand, frag_smile=frag, end_prob=end_prob)

            # generating 3d structure from smiles
            try:
                mol = Chem.MolFromSmiles(new_lig)
                mol = self.generateRdkitConformer(mol)
            except:
                continue

            if mol is None:
                continue

            if old_ligand is None or old_ligand == '':
                # if current/new fragment is the first fragment then place this fragment at initial position
                mol = self.placeRdkitMolAtNewPoint(mol, 0, initial_position[0], initial_position[1], initial_position[2])
            else:
                # else align new/current mol to old molecule
                AllChem.AlignMol(mol, old_ligand_3d, atomMap=atomMap)
                


            # checking if ligand is present inside grid box, if not then try other fragment
            if self.isLigInGridbox(mol) is False:
                continue

            
            # optimizing ligand position
            pdbqt = self.rdkitToPdbqt(mol)
            self.vina.set_ligand_from_string(pdbqt)
            vina_score = self.vina.optimize()[0]

            temp_file_path = os.path.join(self.temp_folder_path, str(uuid.uuid4()))
            self.vina.write_pose(temp_file_path, overwrite=True)
            with open(temp_file_path) as f:
                old_pd = f.read()
            
            # updating coordinates of mol 
            mol = self.updateMolCoordsFromPdbqt(mol, temp_file_path)
            
            os.remove(temp_file_path)
            
            # optimize process or pdbqt to mol can create wrong molecule # fix this in future
            if mol is None:
                continue
                
            
            # checking if ligand is present inside grid box, if not then try other fragment
            if self.isLigInGridbox(mol) is False:
               continue
            
            # calculationg score of new/current ligand using vina
            sa_score = self.getSAScore(mol) - 10
            new_score = vina_score_weight*vina_score + (1-vina_score_weight)*sa_score
            del_score = new_score - old_score
            new_Temp = temp * (alpha**iter)

            if del_score <= 0 or random.random() < math.exp(-del_score / new_Temp):
                new_score = new_score
                detail = {
                    #"out_pdbqt": self.rdkitToPdbqt(mol),
                    "out_sdf": Chem.MolToMolBlock(mol),
                    "added_frag": frag,
                    "in_ligand": old_ligand,
                    "out_ligand": new_lig,
                    "total_score": new_score,
                    "vina_score": vina_score,
                    "sa_score": sa_score+10,
                }

                details.append(detail)
                
                if Descriptors.MolWt(mol) >= max_mw:
                    return mol, details
                else:
                    result, details = self._simulatedAnnealing(old_ligand=new_lig, 
                                                              old_score=new_score, 
                                                              old_ligand_3d=mol, 
                                                              initial_position=initial_position, 
                                                              max_mw=max_mw, 
                                                              temp=temp, 
                                                              iter=iter+1,
                                                              alpha=alpha,
                                                              end_prob=end_prob,
                                                              max_iter_at_state=max_iter_at_state,
                                                              vina_score_weight= vina_score_weight,
                                                              details= details)
                    if result is not None:
                        
                        return result, details
                    
                details.pop()
            else: self.total_frag_rejected_mpc += 1
        return None, details




    def simulatedAnnealing(self, 
                           max_mw: float=100, 
                           temp: float=3000,
                           initial_building_position: list = [0, 0, 0],
                           start_score: float=0,  
                           end_prob:float = 0,
                           vina_score_weight: float = 0.5,
                           ligand:str='', 
                           ligand_3d=None, 
                           alpha:float= 0.9,
                           max_iter_at_state: int= 10,
                           ):
        '''
        end_prob = probablity by which fragment will get added to ends of ligand chain
        vina_score_weight = between [0 to 1]
        max_mw = Maximum molecular weight of fragment
        alpha = rate at which cooling schedule changes
        ligand = smiles of initial ligand
        ligand_3d = rdkit mol object having defined coordinates(lignad_3d smiles should be same as ligand)
        initial_building_position = position at which program will start the building of fragment(if 3d ligand is present then it will be skipped)
        max_iter_at_state = number of failed fragments should be tried at a stage before rejectiong that state 
        '''
        
        if ligand is None: ligand = ""
        if ligand_3d is None and ligand != "": 
            mol = Chem.MolFromSmiles(ligand)
            mol = self.generateRdkitConformer(mol)

            if mol is None:
                raise Exception("Cannot generate 3d structure for initial ligand: ", ligand)
        
        self.total_frag_screened = 0
        self.total_frag_rejected_mpc = 0
        result, details = self._simulatedAnnealing(old_ligand=ligand, 
                                                old_score=start_score, 
                                                old_ligand_3d=ligand_3d, 
                                                initial_position=initial_building_position, 
                                                max_mw=max_mw, 
                                                temp=temp, 
                                                iter=0, 
                                                alpha=alpha,
                                                end_prob=end_prob,
                                                max_iter_at_state=max_iter_at_state,
                                                vina_score_weight=vina_score_weight)
        details = {
            'total_frag_screened': self.total_frag_screened,
            'total_frag_rejected_mpc': self.total_frag_rejected_mpc,
            'state_details': details
        }
        return result, details

