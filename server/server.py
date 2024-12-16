from flask import Flask, request, send_from_directory
import os
import uuid
from multiprocessing import Process
from flask_cors import CORS
from flask import Response, json
import sys
import time
import subprocess
sys.path.append("..")
import generate_ligands

# Setup Flask App
app = Flask(__name__)
CORS(app)

def run_liggen(
    base_dir,
    count,
    grid_center,
    grid_size,
    threads,
    rnn_device,
    alpha,
    chain_extend_probability,
    weight,
    temp,
    score,
    vina_weight):

    target_path = os.path.join(base_dir, 'target.pdbqt')
    output_dir = os.path.join(base_dir, "ligands")
    os.mkdir(output_dir)
    try:
        generate_ligands.mp_pipeline(
            fragment_path = "../frags/ligbuilder_frags.smiles",
            target_path = target_path, 
            output_dir = output_dir, 
            initial_point = grid_center,
            count = count,
            grid_center = grid_center,
            grid_size = grid_size,
            threads = threads,
            rnn_device = rnn_device,
            alpha = alpha,
            chain_extend_probablity = chain_extend_probability,
            weight = weight,
            max_iter = 30,
            temp = temp,
            score = score,
            vina_weight = vina_weight,
            dock=False, 
            rnn=False,
            save_details = True,
            rnn_max_len=100, 
            rnn_count=10,
            rnn_params = '../ligbuilder_model.pt'
        )
    finally:
        with open(os.path.join(base_dir, 'status.txt'), 'w') as f:
            if len(os.listdir(output_dir)) == 0:
                f.write("error")
            else:
                f.write('generated')


@app.route("/submit-job", methods=["POST"])
def submit_job():
    data = request.get_json()
    target_pdbqt = data['target']
    count = data['count']
    grid_center = data['grid_center']
    grid_size = data['grid_size']
    threads = data['threads']
    rnn_device = data['rnn_device']
    alpha = data['alpha']
    chain_extend_probability = data['chain_extend_probability']
    weight = data['weight']
    temp = data['temp']
    score = data['score']
    vina_weight = data['vina_weight']
    
    job_id = str(uuid.uuid4())
    
    if os.path.exists("computation") == False:
        os.mkdir("computation")

    base_dir = os.path.join("computation", job_id)
    os.mkdir(base_dir)

    target_path = os.path.join(base_dir, 'target.pdbqt')
    with open(target_path, 'w') as f:
        f.write(target_pdbqt)
    
    with open(os.path.join(base_dir, 'details.txt')):
        json.dump({
            'count': count,
            'status': 'Generating'
        }, f)
    
    pro = Process(target=run_liggen, args=(base_dir,
        count,
        grid_center,
        grid_size,
        threads,
        rnn_device,
        alpha,
        chain_extend_probability,
        weight,
        temp,
        score,
        vina_weight))
    
    pro.start()

    return Response(json.dumps({'job_id': job_id}))

@app.route("/job-status/<job_id>", methods=["GET"])
def job_status(job_id):

    base_dir = os.path.join("computation", job_id)
    
    if os.path.exists(base_dir):
        if os.path.exists(os.path.join(base_dir, "status.txt")) == True:
            with open(os.path.join(base_dir, "status.txt")) as f:
                return Response(json.dumps({'status': f.read()}))
        else:
            return Response(json.dumps({'status': "generating"}))
    else:
        return Response(json.dumps({'status': "invalid job id"}))
   
@app.route("/download/<job_id>", methods=["GET"])
def download(job_id):
    base_dir = os.path.join("computation", job_id)
    ligands_path = os.path.join(base_dir, 'ligands')
    zip_path = os.path.join(base_dir, "ligands.zip")

    subprocess.run(['zip', '-r', zip_path, ligands_path])
    return send_from_directory(base_dir, "ligands.zip")

if __name__ == "__main__":

    app.run(port=5000)
