import React, { useState } from "react";
import OtherTools from "./otherTools";
import exampleFileContent from "./ExampleFile";
import {
  AppBar,
  TextField,
  Checkbox,
  MenuItem,
  Button,
  Typography,
  Select,
  FormControl,
  FormControlLabel,
  InputLabel,
  Stack,
  Box,
  Container,
  Paper,
  Table,
  TableHead,
  TableRow,
  TableCell,
  TableBody,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  Tabs,
  Tab,
  Toolbar,
  TableContainer
} from "@mui/material";
import axios from "axios";




var ADDRESS = "https://neurocare-liggen.iiitd.edu.in/"
//var ADDRESS = "http://localhost:3000"

const App = () => {
  const [jobId, setJobId] = useState("");
  const [status, setStatus] = useState("");
  const [file, setFile] = useState(null);
  const [helpOpen, setHelpOpen] = useState(false);
  const [activeTab, setActiveTab] = useState(0);


  const handleTabChange = (event, newValue) => {
    setActiveTab(newValue);
  };

  const handleHelpOpen = () => {
    setHelpOpen(true);
  };

  const handleHelpClose = () => {
    setHelpOpen(false);
  };


  const [parameters, setParameters] = useState({
    count: 10,
    grid_center: "",
    grid_size: "",
    threads: 1,
    rnn_device: "gpu",
    alpha: 0.3,
    chain_extend_probability: 0.8,
    weight: 500,
    temp: 50,
    score: 0,
    vina_weight: 0.5,
    jobId: "",
    example: false
  });

  const [ligandDetailsDict, setLigandDetailsDict] = useState(null);

  // Fetch ligand details from server
  const fetchLigandDetails = async (jobId) => {
    try {
      const response = await axios.get(`${ADDRESS}/ligand/details/${parameters.jobId}`);
      setLigandDetailsDict(response.data);
      console.log(response.data['0']['img'])

    } catch (error) {
      console.error("Error fetching ligand details:", error);
    }
  };

  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;
    setParameters({
      ...parameters,
      [name]: type === "checkbox" ? checked : value,
    });
  };

  const submitJob = async () => {
    let rnn_device = parameters.rnn_device;
    if (rnn_device === "cpu") {
      rnn_device = "cpu";
    } else {
      rnn_device = "cuda";
    }

    try {
      let content = {
        count: +parameters.count,
        grid_center: parameters.grid_center.replaceAll(" ", "").split(',').map(str => parseFloat(str)),
        grid_size: parameters.grid_size.replaceAll(" ", "").split(',').map(str => parseFloat(str)),
        threads: +parameters.threads,
        rnn_device: rnn_device,
        alpha: parseFloat(parameters.alpha),
        chain_extend_probability: parseFloat(parameters.chain_extend_probability),
        weight: parseFloat(parameters.weight),
        temp: parseFloat(parameters.temp),
        score: parseFloat(parameters.score),
        vina_weight: parseFloat(parameters.vina_weight),
        target: file.content
      }


      const response = await axios.post(ADDRESS + "/submit-job", content);
      setJobId(response.data.job_id);
    } catch (error) {
      console.error("Error submitting job:", error);
    }
  };

  const checkStatus = async () => {
    if (!parameters.jobId) return;
    try {
      const response = await axios.get(`${ADDRESS}/job-status/${parameters.jobId}`);
      setStatus(response.data.status);
      console.log("sdssfdsfdsfsdfsdfsdfds")
      fetchLigandDetails();
    } catch (error) {
      console.error("Error checking job status:", error);
    }
  };

  const handleDownload = () => {

    if (!parameters.jobId) return;
    // Construct the URL with the job ID
    const downloadUrl = `${ADDRESS}/download/${parameters.jobId}`;

    // Create an anchor element to programmatically trigger a download
    const link = document.createElement("a");
    link.href = downloadUrl;
    link.download = `ligands_${parameters.jobId}.zip`; // Suggested file name
    link.click();
  };

  // Handle file upload
  const handleFileChange = (event) => {
    const uploadedFile = event.target.files[0];

    const reader = new FileReader();
    reader.onload = (e) => {
      const content = e.target.result;
      setFile({
        file_name: uploadedFile.name,
        content: content
      });
    };

    reader.readAsText(uploadedFile);

  };

  const loadExampleJob = async () => {

    // 1. Pre-fill the parameters so the user can see what inputs generate these results
    setParameters({
      count: 10,
      grid_center: "-4, -4, 30", // Replace with your actual 7D9O coordinates
      grid_size: "30, 30, 30",   // Replace with your actual 7D9O grid size
      threads: 1,
      rnn_device: "gpu",
      alpha: 0.3,
      chain_extend_probability: 0.8,
      weight: 500,
      temp: 50,
      score: 0,
      vina_weight: 0.5,
      example: true
    });

    setFile({
      file_name: "example_2g94.pdbqt",
      content: exampleFileContent
    });

  };

  const submitJobUI = () => {
    return (<Container>
      <Box sx={{ textAlign: "center", mt: 4 }}>

        <Typography variant="subtitle1" color="textSecondary">
          Submit jobs for ligand generation.
        </Typography>
      </Box>

      <Box sx={{ mt: 4 }}>
        <Stack direction="row" justifyContent="space-between">
          <Typography variant="h5" gutterBottom>
            Job Parameters
          </Typography>
          <Stack direction="row" spacing={2}>
            <Button
              variant="outlined"
              color="secondary"
              onClick={handleHelpOpen}
            >
              Help
            </Button>


            <Button
              variant="outlined"
              color="secondary"
              onClick={loadExampleJob}
            >
              Example
            </Button>
          </Stack>


        </Stack>
        <br />
        <Stack spacing={3}>
          <Button
            variant="contained"
            component="label"
            sx={{ textAlign: "center" }}
          >
            Upload Target Protein PDBQT
            <input
              type="file"
              hidden
              onChange={handleFileChange}
              accept=".pdbqt"
            />

          </Button>
          {file && <Typography>Target File: {file.file_name}</Typography>}
          <Stack direction="row" spacing={2}>
            <TextField
              fullWidth
              label="Grid Center x, y, z"
              name="grid_center"
              value={parameters.grid_center}
              onChange={handleChange}
            />
            <TextField
              fullWidth
              label="Grid Size x, y, z"
              name="grid_size"
              value={parameters.grid_size}
              onChange={handleChange}
            />
          </Stack>

          <Stack direction="row" spacing={2}>
            <TextField
              fullWidth
              type="number"
              label="Weight of Ligands"
              name="weight"
              value={parameters.weight}
              onChange={handleChange}
            />
          </Stack>

          <Stack direction="row" spacing={2}>
            <TextField
              fullWidth
              type="number"
              label="Number of Ligands"
              name="count"
              value={parameters.count}
              onChange={handleChange}
            />
            <TextField
              fullWidth
              type="number"
              label="Threads"
              name="threads"
              value={parameters.threads}
              onChange={handleChange}
            />
          </Stack>


          <Stack direction="row" spacing={2}>
            <TextField
              fullWidth
              type="number"
              label="Temperature"
              name="temp"
              value={parameters.temp}
              onChange={handleChange}
            />
            <FormControl fullWidth>
              <InputLabel>Device</InputLabel>
              <Select
                name="rnn_device"
                value={parameters.rnn_device}
                onChange={handleChange}
              >
                <MenuItem value="cpu">CPU</MenuItem>
                <MenuItem value="gpu">GPU</MenuItem>
              </Select>
            </FormControl>
          </Stack>

          <Stack direction="row" spacing={2}>
            <TextField
              fullWidth
              type="number"
              step="0.1"
              label="Vina Weight"
              name="vina_weight"
              value={parameters.vina_weight}
              onChange={handleChange}
            />
            <TextField
              fullWidth
              type="number"
              step="0.1"
              label="Alpha"
              name="alpha"
              value={parameters.alpha}
              onChange={handleChange}
            />
          </Stack>

          <Stack direction="row" spacing={2}>
            <TextField
              fullWidth
              type="number"
              step="0.1"
              label="Chain Extend Probability"
              name="chain_extend_probability"
              value={parameters.chain_extend_probability}
              onChange={handleChange}
            />
            <TextField
              fullWidth
              type="number"
              step="1"
              label="Score"
              name="score"
              value={parameters.score}
              onChange={handleChange}
            />
          </Stack>
        </Stack>
        <br />
        <Stack>
          <Button
            variant="contained"
            color="primary"
            onClick={submitJob}
            sx={{ mr: 2 }}
          >
            Submit Job
          </Button>

        </Stack>
      </Box>


      <Dialog open={helpOpen} onClose={handleHelpClose} fullWidth
        maxWidth="md">
        <DialogTitle>Parameter Descriptions</DialogTitle>
        <DialogContent>
          <Typography variant="body1" gutterBottom>
            <strong>Grid Center x, y, z:</strong> Coordinates of the grid center for ligand generation.
          </Typography>
          <Typography variant="body1" gutterBottom>
            <strong>Grid Size x, y, z:</strong> Dimensions of the grid for ligand generation.
          </Typography>
          <Typography variant="body1" gutterBottom>
            <strong>Weight of Ligands:</strong> Weight in atomic mass units.
          </Typography>
          <Typography variant="body1" gutterBottom>
            <strong>Number of Ligands:</strong> Number of ligands to generate.
          </Typography>
          <Typography variant="body1" gutterBottom>
            <strong>Threads:</strong> Number of threads for computation.
          </Typography>
          <Typography variant="body1" gutterBottom>
            <strong>Temperature:</strong> Starting temperature for the Metropolis criteria.
          </Typography>
          <Typography variant="body1" gutterBottom>
            <strong>Device:</strong> Select CPU or GPU for computation.
          </Typography>
          <Typography variant="body1" gutterBottom>
            <strong>Vina Weight:</strong> Weight of vina score between [0 to 1], Final score of ligand = (vina_score_weight)*vina_score + (1-vina_score_weight)*synthesizability_score
          </Typography>
          <Typography variant="body1" gutterBottom>
            <strong>Alpha:</strong> Factor by which cooling schedule (Temperature) changes.
          </Typography>
          <Typography variant="body1" gutterBottom>
            <strong>Chain Extend Probability:</strong> Probablity by which fragment will get added to ends of ligand.
          </Typography>
          <Typography variant="body1" gutterBottom>
            <strong>Score:</strong> Initial minimum score to accept a ligand.
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={handleHelpClose} color="primary">
            Close
          </Button>
        </DialogActions>
      </Dialog>
    </Container>);
  }

  const checkStatusUI = () => {
    return (<Container sx={{ mt: 4 }}>
      <Stack direction="row" spacing={2}>
        <Button
          variant="outlined"
          color="secondary"
          onClick={checkStatus}
        >
          Check Status
        </Button>

        <Button
          variant="outlined"
          color="secondary"
          onClick={handleDownload}
        >
          Download Ligands
        </Button>

        <TextField
          fullWidth
          label="Job Id"
          name="jobId"
          value={parameters.jobId}
          onChange={handleChange}
        />

      </Stack>

    </Container>);
  }


  // Render ligand details
  const ligandDetailsUI = (ligandDetails, key) => {
    console.log(ligandDetails);
    if (!ligandDetails || Object.keys(ligandDetails).length === 0) return (
      <Box sx={{ mt: 4 }}>
        <Typography variant="h5" gutterBottom>
          Ligand {key} : Generating
        </Typography>
      </Box>
    );

    return (
      <Box sx={{ mt: 4 }}>
        <Typography variant="h5" gutterBottom>
          Ligand {key}
        </Typography>

        {/* General Details */}
        <Paper sx={{ p: 2, mb: 4 }}>

          {ligandDetails.img ? <Stack direction="row" spacing={2}>
            {<img
              src={`data:image/png;base64,${ligandDetails.img}`}
              alt="Ligand"
              style={{ border: "1px solid #ccc" }}
            />}

            <Box>


              <Typography>
                <strong>Vina Score:</strong>{" "}
                {ligandDetails.undocked_final_energy}
              </Typography>
              <Typography>
                <strong>Synthesizability Score:</strong>{" "}
                {ligandDetails.synthesizability_score}
              </Typography>

            </Box>

          </Stack> : <Box> Generating </Box>}


        </Paper>

        {/* State Details Table */}
        <Table>
          <TableHead>
            <TableRow>
              <TableCell>Step</TableCell>
              <TableCell>Added Fragment</TableCell>
              <TableCell>Sub Ligand</TableCell>
              <TableCell>Total Score</TableCell>
              <TableCell>Vina Score</TableCell>
              <TableCell>Synthesizability Score</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {ligandDetails.state_details.map((detail, index) => (
              <TableRow key={index}>
                <TableCell>{index + 1}</TableCell>
                <TableCell>{detail.added_frag}</TableCell>
                <TableCell>{detail.out_ligand}</TableCell>
                <TableCell>{detail.total_score}</TableCell>
                <TableCell>{detail.vina_score}</TableCell>
                <TableCell>{detail.sa_score}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </Box>
    );
  };



  const introductionUI = () => (
    <Container maxWidth="md">
      {/* Header */}
      <Box sx={{ marginBottom: 4, textAlign: "center" }}>
        <Typography variant="h4" component="h1" gutterBottom>
          LigGen - A GEN-AI and Monte Carlo Simulated Annealing Based
          De Novo Drug Design Toolbox
        </Typography>
      </Box>

      {/* Introduction */}
      <Box sx={{ marginBottom: 3 }}>
        <Typography variant="body1" paragraph>
          We have developed a novel Gen-AI and Monte Carlo Simulated Annealing
          based de novo ligand generation tool referred to as <strong>LigGen</strong>, which
          uses fragments in SMILES format as building blocks.
        </Typography>
        <Typography variant="body1" paragraph>
          When compared to other de novo drug discovery packages such as
          <strong> LigBuilder</strong> and <strong>Pocket2Mol</strong>, LigGen's fragment generation
          is based on generative AI exploiting an RNN-LSTM approach. This approach
          was pretrained on the <strong>ChEMBL dataset</strong> and fine-tuned on LigBuilder’s
          fragment library.
        </Typography>
      </Box>

      {/* Performance Highlights */}
      <Box sx={{ marginBottom: 3 }}>
        <Typography variant="h5" component="h2" gutterBottom>
          Performance Highlights
        </Typography>
        <Typography variant="body1" paragraph>
          LigGen’s performance was evaluated on the <strong>CrossDock dataset</strong> and
          three Alzheimer’s-related proteins to compare it with several
          state-of-the-art de novo drug design tools, including
          <strong> Pocket2Mol</strong>, <strong>LigBuilder V3</strong>, and others.
        </Typography>
        <Typography variant="body1" paragraph>
          On the CrossDock dataset, LigGen demonstrated competitive binding
          affinities, achieving a Vina score of <strong>-7.508 ± 1.83</strong>,
          outperforming baseline methods like <strong>3D-SBDD</strong>, <strong>FLAG</strong>, and
          <strong>DrugGPS</strong>. LigGen also showcased strong diversity in the generated
          molecules while maintaining reasonable synthesizability.
        </Typography>
        <Typography variant="body1" paragraph>
          Although tools like Pocket2Mol showed higher QED values, LigGen
          consistently produced ligands with good <strong>Lipinski compliance</strong>,
          suggesting its balance between drug-likeness and innovation.
        </Typography>
      </Box>

      {/* Alzheimer's Protein Analysis */}
      <Box sx={{ marginBottom: 3 }}>
        <Typography variant="h5" component="h2" gutterBottom>
          Alzheimer’s-Related Protein Analysis
        </Typography>
        <Typography variant="body1" paragraph>
          In the Alzheimer’s-related protein analysis, ligands generated by LigGen
          consistently showed better binding affinities compared to those from
          LigBuilder and Pocket2Mol. LigGen demonstrated a balanced approach
          between ligand synthesizability and structural complexity, as seen in
          its ability to produce ligands with moderate synthesizability scores.
        </Typography>
      </Box>

      {/* Conclusion */}
      <Box sx={{ marginBottom: 3 }}>
        <Typography variant="h5" component="h2" gutterBottom>
          Conclusion
        </Typography>
        <Typography variant="body1" paragraph>
          Overall, the results indicate that LigGen is a robust tool for fragment-based
          ligand generation, capable of producing high-quality ligands with
          competitive docking scores and favorable synthesizability. LigGen provides
          a novel method for generating ligands using a fragment-based de novo
          drug design approach.
        </Typography>
      </Box>
    </Container>
  );

  const otherToolsUI = () => (
    <Container>
      <Typography variant="h4" gutterBottom>
        Other Tools
      </Typography>
      <Typography variant="body1">Explore additional functionalities and tools here...</Typography>
    </Container>
  );

  const computationUi = () => {



    return (
      <Container maxWidth="md">

        {submitJobUI()}
        <Container>
          <Box sx={{ mt: 4 }}>
            <Typography variant="subtitle1">Job ID: {jobId}</Typography>
          </Box>
        </Container>

        {checkStatusUI()}

        <Typography variant="subtitle2">Status: {status}</Typography>

        <Box>

          {ligandDetailsDict && Object.keys(ligandDetailsDict).length > 0 ? (
            Object.keys(ligandDetailsDict)
              .map((key) => Number(key)) // Convert keys to numbers for numeric sorting
              .sort((a, b) => a - b) // Sort numerically in ascending order
              .map((key) => {
                const ligandDetails = ligandDetailsDict[key.toString()]; // Access value using sorted key
                return ligandDetailsUI(ligandDetails, key); // Pass key if needed
              })
          ) : (
            <Typography variant="h6" align="center" sx={{ mt: 4 }}>
              No Ligand Details Available
            </Typography>
          )}


        </Box>
      </Container >
    );
  }

  const ourResultsUI = () => (
    <Container>
      <Typography variant="h4" gutterBottom>
        Results: CrossDock Dataset
      </Typography>
      <Typography variant="body1" gutterBottom>
        We evaluated LigGen using the test set of the CrossDock dataset. The test set contains 100 diverse protein-ligand pairs. This table summarizes the molecular properties of the ligands generated for these targets. We compared LigGen against multiple baselines, including 3D-SBDD, Pocket2Mol, GraphBP, TargetDiff, DecompDiff, DiffSBDD, FLAG, DrugGPS, Lingo3DMol, and Frag2Seq. Baseline results are sourced from the paper “Fragment and Geometry Aware Tokenization of Molecules for Structure-Based Drug Design Using Language Models”.
        We use the same evaluation matrix which was used in baselines. Vina Score estimates the binding affinity between generated molecules and given protein pockets; QED is a measure used to assess the drug-likeness of a molecule based on its molecular properties; SA estimates how easy it would be to synthesize a given chemical compound; Lipinski measures how well a molecule satisfies the Lipinski's rule of five , which evaluates the drug-likeness of a molecule; Diversity measures the average pairwise diversity (calculated as 1-Tanimotosimilarity) of generated molecules for a binding pocket; Time is the average time cost to generate 100 molecules for a protein pocket in the test set. All the Vina scores are calculated by QVina, and the chemical properties are calculated by RDKit .
      </Typography>
      <TableContainer component={Paper}>
        <Table>
          <TableHead>
            <TableRow>
              <TableCell><strong>Methods</strong></TableCell>
              <TableCell><strong>Vina Score (↓)</strong></TableCell>
              <TableCell><strong>QED (↑)</strong></TableCell>
              <TableCell><strong>SA (↑)</strong></TableCell>
              <TableCell><strong>Lipinski (↑)</strong></TableCell>
              <TableCell><strong>Diversity (↑)</strong></TableCell>
              <TableCell><strong>Time (s, ↓)</strong></TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {[
              ["Test set*", "-6.87±2.32", "0.47±0.20", "0.72±0.14", "4.34±1.14", "-", "-"],
              ["3D-SBDD*", "-5.88±1.91", "0.50±0.17", "0.67±0.14", "4.78±0.51", "0.74±0.09", "15986.4±9851.0"],
              ["Pocket2Mol*", "-7.05±2.80", "0.57±0.16", "0.75±0.12", "4.93±0.27", "0.73±0.15", "2827.3±1456.8"],
              ["GraphBP*", "-4.71±4.03", "0.50±0.12", "0.30±0.09", "4.88±0.37", "0.84±0.01", "1162.8±438.5"],
              ["TargetDiff*", "-7.31±2.47", "0.48±0.20", "0.58±0.13", "4.59±0.83", "0.71±0.09", "~3428"],
              ["DecompDiff*", "-6.60±2.11", "0.49±0.21", "0.65±0.14", "4.49±1.02", "0.72±0.10", "~6189"],
              ["DiffSBDD*", "-7.17±3.28", "0.55±0.20", "0.72±0.12", "4.74±0.59", "0.71±0.07", "629.9±277.2"],
              ["FLAG*", "-6.38±3.24", "0.48±0.19", "0.70±0.15", "4.65±0.74", "0.70±0.14", "1289.1±378.0"],
              ["DrugGPS*", "-6.60±2.38", "0.46±0.21", "0.62±0.15", "4.49±0.99", "0.73±0.10", "1007.8±554.1"],
              ["Lingo3DMol*", "-7.25±1.69", "0.26±0.15", "0.65±0.08", "3.12±1.25", "0.48±0.12", "1481.9±1512.8"],
              ["Frag2Seq*", "-7.36±1.96", "0.64±0.15", "0.64±0.11", "4.98±0.11", "0.71±0.07", "48.8±14.6"],
              ["LigGen", "-7.50±1.83", "0.43±0.22", "0.68±0.10", "4.60±0.73", "0.69±0.13", "1271.335±3107.77"],
            ].map((row, index) => (
              <TableRow key={index}>
                {row.map((cell, cellIndex) => (
                  <TableCell key={cellIndex}>{cell}</TableCell>
                ))}
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>
      <br /> <br /><br /> <br /> <br />
      <Typography variant="h4" gutterBottom>
        Alzheimer’s-Related Protein Analysis
      </Typography>
      <Typography>
        In this section, we compare the performance of LigGen by generating ligands for Alzheimer's-Related Targets Beta-secretase 1, Monoamine Oxidase B and Acetylcholinesterase against tools LigBuilder V3 and Pocket2Mol. LigBuilder V3 and Pocket2Mol are popular ligand-building tools based on Genetic Algorithms and Deep Learning, respectively. We have generated results for three different targets with PDB IDs 2G94, 2V5Z, and 7D9O. The designed ligands were evaluated based on their Binding Affinities (using AutoDock Vina).
        A total of 100 ligands were generated for each protein using all three programs, LigGen, Pocket2Mol and LigBuilder. Subsequently, these compounds underwent docking using AutoDock-Vina.
      </Typography>
      <div style={{ display: "flex", justifyContent: "center", gap: "20px", alignItems: "center" }}>
        <div>
          <img src="/images/ba_7d9o.png" alt="7D9O: Docking Scores" style={{ width: "350px", height: "auto" }} />
        </div>
        <div>
          <img src="/images/ba_2v5z.png" alt="2V5Z: Docking Scores" style={{ width: "350px", height: "auto" }} />
        </div>
        <div>
          <img src="/images/ba_2g94.png" alt="2G94: Docking Scores" style={{ width: "350px", height: "auto" }} />
        </div>
      </div>

    </Container>
  );

  return (
    <Box>
      <AppBar position="static">
        <Toolbar>
          {/* Logo or Title */}
          <Box display="flex" alignItems="center" sx={{ marginRight: 2 }}>
            <Typography variant="h6" noWrap component="div">
              LigGen Webserver
            </Typography>
          </Box>

          <Tabs value={activeTab} onChange={handleTabChange} indicatorColor="secondary"
            textColor="inherit"
            centered sx={{ flexGrow: 0.5 }} >
            <Tab label="Introduction" />
            <Tab label="Computation" />
            <Tab label="Our Results" />
            <Tab label="Other Tools" />
          </Tabs>
        </Toolbar>
      </AppBar>
      <Box sx={{ p: 3 }}>
        {activeTab === 0 && introductionUI()}
        {activeTab === 1 && computationUi()}
        {activeTab === 2 && ourResultsUI()}
        {activeTab === 3 && OtherTools()}
      </Box>
    </Box>
  );
};


export default App;
