import argparse
import utils as utils
from mcsa import SimulatedAnnealing
from vina import Vina
from concurrent.futures import ProcessPoolExecutor
import concurrent
import os 
import time
import json
import math
import torch
import argparse
from rnn_selfies import RNNSelfies
from rnn_config import *
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from selfies_dataset import SelfiesDataset
from torch import nn
import torch.optim as optim
from tqdm import tqdm
from openbabel import pybel as pb
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

def read_smiles(smiles_path):
    with open(smiles_path, 'r') as f:
        frags = f.read().split("\n")
        frags = [f.split()[0] for f in frags if f != '']
        return frags
    
def clean_smiles(smiles_list):
    # removing disconnected smiles
    smiles_list = remove_stereo_symbols([smiles for smiles in smiles_list if '.' not in smiles])

    return smiles_list

def remove_stereo_symbols(smiles_list):
    smiles_list = [smile.replace("\\", "").replace("/", "") for smile in smiles_list]

    return smiles_list

def _pipeline(fragments:list[str], target_path, output_dirs, initial_point, grid_center, 
             grid_size, initial_ligand:str=None, chain_extend_probablity=0.2, weight=500, 
             max_iter=100, temp=100, score=0, vina_weight=0.5, alpha=0.2, dock=False, save_details:bool=False):

    vina = Vina(cpu=1)
    sa = SimulatedAnnealing(fragments=fragments, vina=vina)
    # setting target and computing grid box
    sa.setTarget(target_pdbqt_path=target_path, grid_param=(None, grid_center, grid_size))

    for output_dir in output_dirs:
        if os.path.exists(output_dir)==False or os.path.isdir(output_dir)==False:
            os.mkdir(output_dir)
            
        try:
            result = sa.simulatedAnnealing(
                    max_mw=weight,
                    temp = temp,
                    initial_building_position=initial_point,
                    start_score=score,
                    end_prob=chain_extend_probablity,
                    vina_score_weight=vina_weight,
                    ligand=initial_ligand,
                    max_iter_at_state=max_iter,
                    alpha=alpha
                )
            
            if result is None or len(result)==0 or result[0] is None:
                print("Failed to generate ligand")
                continue
            
            final_ligand_sdf = result[1]["state_details"][-1]["out_sdf"]
            details = result[1]
            
            # saving final ligand without docking
            with open(os.path.join(output_dir, 'ligand.sdf'), 'w') as f:
                f.write(final_ligand_sdf)

            if save_details == True:
                # saving states
                if os.path.exists(os.path.join(output_dir, 'state')) == False:
                    os.mkdir(os.path.join(output_dir, 'state'))

                for i in range(len(details['state_details'])):
                    sdf = details['state_details'][i]['out_sdf']
                    del details['state_details'][i]['out_sdf']
                    with open(os.path.join(output_dir, 'state', str(i)+'.sdf'), 'w') as f:
                        f.write(sdf)


                undocked_energy = vina.score()[0]

                # saving details
                details['undocked_final_energy'] = undocked_energy
                details['synthesizability_score'] = details['state_details'][-1]['sa_score']

                with open(os.path.join(output_dir, 'details.txt'), 'w') as f:
                    json.dump(details, f, indent=4)

            if dock is True:
                pymol = pb.readstring(format="sdf", string=final_ligand_sdf)
                pymol.addh()
                _ = pymol.calccharges("gasteiger")
                pdbqt = pymol.write(format="pdbqt")

                # docking final ligand
                vina.set_ligand_from_string(pdbqt)
                vina.dock(n_poses=1)
                docked_energy = vina.energies()[0][0]
                vina.write_poses(os.path.join(os.path.join(output_dir, 'docked_ligand.pdbqt')), n_poses=1, overwrite=True)

                if save_details == True:
                    # saving details again if vina docking do not fails
                    details['docked_final_energy'] = docked_energy

                    with open(os.path.join(output_dir, 'details.txt'), 'w') as f:
                        json.dump(details, f, indent=4)

        except Exception as e:
            print(e)


def generate_fragments(parms, device, count=256, max_len=40):
    """this method returns total of 1024 fragments"""

    vocab = get_vocab()
    model = RNNSelfies(vocab_size=len(vocab),
        embed_dim=256,
        hidden_size=512,
        num_layers=3,
        dropout=0)
    
    model.load_state_dict(torch.load(parms, map_location=device))
    model = model.to(device)

    batch_size = 32
    iteration = math.ceil(count/batch_size)
    pbar = tqdm(total=iteration*batch_size)
    
    smiles_list = []
    for i in range(iteration):
        smiles_list += model.generateSmiles(batch_size=batch_size, vocab=vocab, max_len=max_len)
        pbar.update(batch_size)
    
    pbar.close()

    return smiles_list

def mp_pipeline(fragment_path, target_path, output_dir, initial_point, grid_center, 
             grid_size, count=1, threads=1, initial_ligand:str=None, chain_extend_probablity=0.2, weight=500, 
             max_iter=10, temp=300, score=0, vina_weight=0.5, alpha=0.9, dock=False, rnn=False, rnn_params=None, 
             rnn_device="cpu", rnn_max_len=40, rnn_count=256, save_details:bool=False):
    
    fragments = None

    if rnn == False:
        # reading fragments 
        fragments = read_smiles(fragment_path)
        
        # cleaning fragments
        fragments = clean_smiles(fragments)

    output_lig_dirs = [os.path.join(output_dir, str(i)) for i in range(count)]
    
    # splitting output dirs for assigning them to multi-processes
    splitted_dir = [output_lig_dirs[i*(count//threads):(i+1)*(count//threads)] for i in range(threads)]

    # adding lefts
    lefts = output_lig_dirs[(count//threads)*threads: len(output_lig_dirs)]
    for i in range(len(lefts)):
        splitted_dir[i].append(lefts[i])

    # starting process pool
    with ProcessPoolExecutor(max_workers=threads) as exe:
        try:
            if rnn == False:
                futures = [exe.submit(
                    _pipeline,
                    fragments, target_path, lig_dirs, initial_point, grid_center, 
                    grid_size, initial_ligand, chain_extend_probablity, weight, 
                    max_iter, temp, score, vina_weight, alpha, dock,save_details
                ) for lig_dirs in splitted_dir]
            else :
                futures = [exe.submit(
                    _pipeline,
                    clean_smiles(generate_fragments(rnn_params, rnn_device, rnn_count, rnn_max_len)), target_path, lig_dirs, initial_point, grid_center, 
                    grid_size, initial_ligand, chain_extend_probablity, weight, 
                    max_iter, temp, score, vina_weight, alpha, dock,save_details
                ) for lig_dirs in splitted_dir]

        except KeyboardInterrupt:
            exe.shutdown(wait=False, cancel_futures=True)
            print("Terminating")


  

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                        prog='Denovo-frag',
                        description='Generate ligands using fragment',
                        )

    default_model_params = 'model.pt'
    default_device = 'cuda' if torch.cuda.is_available() else 'cpu'

    parser.add_argument('-fp', '--fragment_path', type=str, required=False,
                        help='path of file containing fragments in smile format')

    parser.add_argument('-tp', '--target_path', type=str, required=True,
                        help='target path in pdbqt format')

    parser.add_argument('-o', '--output_dir', type=str, required=True,
                        help='output dir location')

    parser.add_argument('-c', '--count', type=int, required=True,
                        help='number of ligands to generate')

    # parser.add_argument('-ip', '--initial_point', type=utils.parse_cmp_coordinates, required=True,
    #                     help='[x,y,z] -point at which program will start the building of fragment')

    parser.add_argument('-gc', '--grid_center', type=utils.parse_cmp_coordinates, required=True,
                        help='[x,y,z] -center point of grid box')

    parser.add_argument('-gs', '--grid_size', type=utils.parse_cmp_coordinates, required=True,
                        help='[x,y,z] -size of grid box')

    parser.add_argument('-th', '--threads', type=int, default=1, help='number of threads')

    parser.add_argument('-rnn', '--rnn', action='store_true',
                        help='generate fragments using rnn')
    
    parser.add_argument('-p', '--rnn_params', type=str, default=default_model_params,
                        help=f'path of model parameters (default= {default_model_params})')
    
    parser.add_argument('-d', '--rnn_device', type=str, default=default_device,
                        help=f'cpu or gpu (default= {default_device})')

    parser.add_argument('-ml', "--rnn_max_len", type=int, default=40,
                        help=f'max lenght of output generated by rnn (default= {40})')
    
    parser.add_argument('-rc', "--rnn_count", type=int, default=256,
                        help=f'count of fragment rnn will generate for each ligand (default= {256})')
    # parser.add_argument('-il', '--initial_ligand', type=str, default='',
    #                     help='smiles of initial ligand')
    
    parser.add_argument('-al', '--alpha', type=float, default=0.3,
                    help='factor by which cooling schedule changes  default=0.3')

    parser.add_argument('-cp', '--chain_extend_probablity', type=float, default=0.8,
                        help='probablity by which fragment will get added to ends of ligand chain default=0.8')

    parser.add_argument('-w', '--weight', type=float, default=500,
                        help='molecular weight of ligand default=500')

    parser.add_argument('-mi', '--max_iter', type=int, default=50,
                        help='number of failed fragments should be tried at a stage before rejectiong that state default=50')

    parser.add_argument('-t', '--temp', type=float, default=50,
                        help='temperature default=50')

    parser.add_argument('-s', '--score', type=float, default=0,
                        help='initial score default=0')

    parser.add_argument('-vw', '--vina_weight', type=float, default=0.5,
                        help='weight of vina score between [0 to 1], default=0.5, 1-vina_weight = sa_score')

    parser.add_argument('-do', '--dock', action='store_true',
                        help='dock the generated ligand with protien if flag is provided')
    

    parser.add_argument('-de', '--details', action='store_true',
                        help='save the details of each step of ligand generation if flag is provided')

    args = parser.parse_args()

    for k, v in args.__dict__.items():
        print(k, ":", v)

    
    mp_pipeline(fragment_path=args.fragment_path, 
                target_path=args.target_path, 
                output_dir=args.output_dir, 
                initial_point=args.grid_center, 
                grid_center=args.grid_center, 
                grid_size=args.grid_size, 
                count=args.count, 
                threads=args.threads,  
                chain_extend_probablity=args.chain_extend_probablity, 
                weight=args.weight, 
                max_iter=args.max_iter, 
                temp=args.temp, 
                score=args.score, 
                vina_weight=args.vina_weight,
                alpha = args.alpha,
                dock=args.dock,
                rnn = args.rnn,
                rnn_params=args.rnn_params,
                rnn_device=args.rnn_device,
                rnn_max_len = args.rnn_max_len,
                rnn_count = args.rnn_count,
                save_details=args.details)
  
