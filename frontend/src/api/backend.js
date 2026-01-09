import axios from "axios";

const API = axios.create({
  baseURL: "http://localhost:8000", // backend FastAPI URL
});

export default API;
