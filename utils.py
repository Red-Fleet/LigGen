import argparse
from openbabel import openbabel as ob
from openbabel import pybel as pb
import numpy as np
import selfies as sf
from openbabel import openbabel as ob
from openbabel import pybel as pb

def smilesToSelfies(smiles):
    try:
        return sf.encoder(smiles.split()[0])
    except :
        return None

smiles_atoms = {'Al', 'As', 'B', 'Br', 'C', 'Cl', 'F', 'I', 'K', 'Li', 'N',
    'Na', 'O', 'P', 'S', 'Se', 'Si', 'Te', 'se', 'te', 'c', 'n', 'o', 'p', 's'}


def parse_cmp_coordinates(string):
    result = string.replace(' ', '').replace('[', '').replace(']', '').split(',')
    if len(result) != 3: raise argparse.ArgumentTypeError('Incorrect Coordinates')

    for i in range(3):
        try:
            result[i] = float(result[i])
        except:
            raise argparse.ArgumentTypeError('Incorrect Coordinates')

    
    return result



def justifyRingCloserLabelInSmiles(in_smiles, reference_smiles):
    '''return new smiles similar to in_smiles after offsetting ring labels of in_smiles,
    offset is calculated with respect to reference_smiles'''

    # finding all labels present in reference smiles
    ref_lbs = ['']
    for c in reference_smiles:
        if c.isdecimal():
            ref_lbs[-1] += c
        else:
            if len(ref_lbs[-1]) != 0:
                ref_lbs.append('')
    
    ref_lbs = [int(e) for e in ref_lbs if e != '']

    # finding offset
    if len(ref_lbs) == 0:
        offset = 0
    else:
        offset = max(ref_lbs)

    # adding offset to labels of in_smiles
    curr = ''
    out_smiles = ''
    for i, c in enumerate(in_smiles):
        if c.isdecimal():           
            curr += c
        else:
            if curr != '':
                out_smiles += str(int(curr) + offset)
                curr = ''
            out_smiles += c
    if curr != '':
        out_smiles += str(int(curr) + offset)

    return out_smiles


def readDetailsFromPdbLine(line):
    atom = line[0:6].replace(" ", '')
    atom_serial = line[6:11].replace(" ", '')
    atom_name = line[12:16].replace(" ", '')
    alternate_location = line[16].replace(" ", '')
    residue_name = line[17:20].replace(" ", '')
    chain_identifier = line[21].replace(" ", '')
    residue_sequence_number = line[22:26].replace(" ", '')
    code = line[26].replace(" ", '')
    x_coordinate = line[30:38].replace(" ", '')
    y_coordinate = line[38:46].replace(" ", '')
    z_coordinate = line[46:54].replace(" ", '')
    occupancy = line[54:60].replace(" ", '')
    temperature = line[60:66].replace(" ", '')
    segment_identifier = line[72:76].replace(" ", '')
    element_symbol = line[76:78].replace(" ", '')
    charge = line[78:80].replace(" ", '')

    return {
        'atom':atom,
        'atom_serial':atom_serial,
        'atom_name':atom_name,
        'alternate_location':alternate_location,
        'residue_name':residue_name,
        'chain_identifier':chain_identifier,
        'residue_sequence_number':residue_sequence_number,
        'code':code,
        'x_coordinate':x_coordinate,
        'y_coordinate':y_coordinate,
        'z_coordinate':z_coordinate,
        'occupancy':occupancy,
        'temperature':temperature,
        'segment_identifier':segment_identifier,
        'element_symbol':element_symbol,
        'charge': charge
    }

def getStringOfSize(value, size, left_justification = True):
    result = [' ']*size
    value = str(value)[0:size]
    spaces = size - len(value)
    if left_justification == True:
        result[0:size] = value + ' '*spaces
    else:
        result[0:size] = ' '*spaces + value
    
    return ''.join(result)


def getPdbLineFromDetails(
        atom = '',
        atom_serial = '',
        atom_name = '',
        alternate_location = '',
        residue_name = '',
        chain_identifier = '',
        residue_sequence_number = '',
        code = '',
        x_coordinate = '',
        y_coordinate = '',
        z_coordinate = '',
        occupancy = '',
        temperature = '',
        segment_identifier = '',
        element_symbol = '',
        charge = ''
    ):

    atom = getStringOfSize(atom, 6, True)
    atom_serial = getStringOfSize(atom_serial, 5, False)
    atom_name = getStringOfSize(atom_name, 4, False)
    alternate_location = getStringOfSize(alternate_location, 1, True)
    residue_name = getStringOfSize(residue_name, 3, False)
    chain_identifier = getStringOfSize(chain_identifier, 1, True)
    residue_sequence_number = getStringOfSize(residue_sequence_number, 4, False)
    code = getStringOfSize(code, 1, True)
    x_coordinate = getStringOfSize(x_coordinate, 8, False)
    y_coordinate = getStringOfSize(y_coordinate, 8, False)
    z_coordinate = getStringOfSize(z_coordinate, 8, False)
    occupancy = getStringOfSize(occupancy, 6, False)
    temperature = getStringOfSize(temperature, 6, False)
    segment_identifier = getStringOfSize(segment_identifier, 4, True)
    element_symbol = getStringOfSize(element_symbol, 2, False)
    charge = getStringOfSize(charge, 2, True)


    line = atom + atom_serial + ' ' + atom_name + alternate_location + residue_name + ' ' + chain_identifier + residue_sequence_number \
            + code + ' '*3 + x_coordinate + y_coordinate + z_coordinate + occupancy + temperature + ' '*6 + segment_identifier + element_symbol \
            + charge
    
    return line
    

    
def pdbqtToPdb_do_not_use(pdbqt):
    lines = pdbqt.split('\n')

    result = []
    for line in lines:
        if line.startsWith('ATOM') or line.startsWith('HETATM'):
            details = getPdbLineFromDetails(line)
            result.append(getPdbLineFromDetails(atom=details['atom'],
                                                atom_serial=details['atom_serial'],
                                                atom_name=details['atom_name'],
                                                x_coordinate=details['x_coordinate'],
                                                y_coordinate=details['y_coordinate'],
                                                z_coordinate=details['z_coordinate'],
                                                element_symbol=details['atom_name']
                                                )+ '\n')
    
    return ''.join(result)


def getAtomCoordinatesFromMol(mol: ob.OBMol):
    result = []
    for atom in  ob.OBMolAtomIter(mol):
        result.append([atom.GetX(), atom.GetY(), atom.GetZ()])
    
    return np.array(result)


# generate grid box around coordinate
def getGridbox(coordinates: np.array, padding: float, min=None):
    min_vals = coordinates.min(axis=0) - padding
    max_vals = coordinates.max(axis=0) + padding
    center = (min_vals + max_vals)/2
    size = max_vals-min_vals

    if min is not None:
        size = np.array([size, [min]*3]).max(axis=0)
    return center, size

# get grid using the docked ligand
def getGridFromLigand(lig_path, lig_format, min=20, padding=5):
    '''return center and size'''
    lig_pb = next(pb.readfile(lig_format, lig_path))

    return getGridbox(getAtomCoordinatesFromMol(lig_pb.OBMol), padding, min)


def num_atoms_in_smiles(smiles):
    '''return number of atoms present in smiles'''
    total = 0
    i = 0
    while i < len(smiles):
        if smiles[i] in smiles_atoms:
            total += 1
        elif i+1 < len(smiles) and smiles[i:i+2] in smiles:
            total += 1
            i += 1
        i += 1
    
    return total

def refine_smiles(smiles_list, min_atoms, max_atoms):
    '''return new list contaning smiles which have atoms in range of min_atom and max_atom'''

    result = []
    for smiles in smiles_list: 
        count = num_atoms_in_smiles(smiles)

        if count >= min_atoms and count <= max_atoms:
            result.append(smiles)

    return result

def refine_smiles_file(in_file_path, out_file_path, min_atoms, max_atoms):
    with open(in_file_path) as f_in:
        smiles_list = [smiles.split()[0] for smiles in f_in.readlines()]

        refined_smiles_list = refine_smiles(smiles_list, min_atoms, max_atoms)

        with open(out_file_path, 'w') as f_out:
            for smiles in refined_smiles_list:
                f_out.write(smiles + '\n')


def pdbToPdbqt(in_path, out_path):
    m = next(pb.readfile('pdb', in_path))
    m.addh()
    _ = m.calccharges("gasteiger")
    obconv = ob.OBConversion()
    m.write('pdbqt', out_path, overwrite=True, opt={'r':obconv.OUTOPTIONS})