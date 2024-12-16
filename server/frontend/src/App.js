import React, { useState } from "react";
import {
  TextField,
  Checkbox,
  Button,
  Typography,
  Select,
  MenuItem,
  FormControl,
  FormControlLabel,
  InputLabel,
  Stack,
  Box,
  Container,
} from "@mui/material";
import axios from "axios";




const App = () => {
  const [jobId, setJobId] = useState("");
  const [status, setStatus] = useState("");
  const [file, setFile] = useState(null);
  

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
    jobId: ""
  });

  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;
    setParameters({
      ...parameters,
      [name]: type === "checkbox" ? checked : value,
    });
  };

  const submitJob = async () => {

    try {
      let content = {
        count: +parameters.count,
        grid_center: parameters.grid_center.replaceAll(" ", "").split(',').map(str => parseFloat(str)),
        grid_size: parameters.grid_size.replaceAll(" ", "").split(',').map(str => parseFloat(str)),
        threads: +parameters.threads,
        rnn_device: parameters.rnn_device,
        alpha: parseFloat(parameters.alpha),
        chain_extend_probability: parseFloat(parameters.chain_extend_probability),
        weight: parseFloat(parameters.weight),
        temp: parseFloat(parameters.temp),
        score: parseFloat(parameters.score),
        vina_weight: parseFloat(parameters.vina_weight),
        target: file.content
      }
      
      
      const response = await axios.post("http://localhost:5000/submit-job", content);
      setJobId(response.data.job_id);
    } catch (error) {
      console.error("Error submitting job:", error);
    }
  };

  const checkStatus = async () => {
    if (!parameters.jobId) return;
    try {
      const response = await axios.get(`http://localhost:5000/job-status/${parameters.jobId}`);
      setStatus(response.data.status);
    } catch (error) {
      console.error("Error checking job status:", error);
    }
  };

  const handleDownload = () => {

    if (!parameters.jobId) return;
    // Construct the URL with the job ID
    const downloadUrl = `http://localhost:5000/download/${parameters.jobId}`;
    
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

  const submitJobUI = () => {
    return (<Container>
      <Box sx={{ textAlign: "center", mt: 4 }}>
        <Typography variant="h4" gutterBottom>
          LigGen Web Server
        </Typography>
        <Typography variant="subtitle1" color="textSecondary">
          Submit jobs for ligand generation and docking.
        </Typography>
      </Box>
  
      <Box sx={{ mt: 4 }}>
        <Typography variant="h5" gutterBottom>
          Job Parameters
        </Typography>
        <Stack spacing={3}>
          <Button
            variant="contained"
            component="label"
            sx={{ textAlign: "center" }}
          >
            Upload Target File
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
          <Stack direction="row" spacing={2}>
  
            <FormControlLabel
              control={
                <Checkbox
                  checked={parameters.dock}
                  name="dock"
                  onChange={handleChange}
                />
              }
              label="Dock Ligand"
            />
  
          </Stack>
        </Stack>
  
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
    </Container>);
  }

  const checkStatusUI = () =>{
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
    
    </Container >
  );
};

export default App;
