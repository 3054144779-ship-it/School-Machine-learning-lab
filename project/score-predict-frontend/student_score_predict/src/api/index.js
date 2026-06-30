import axios from 'axios'

const api = axios.create({
  baseURL: '/api/prediction',
  timeout: 10000,
})

export function getFeatures() {
  return api.get('/features')
}

export function predict(features) {
  return api.post('/predict', { features })
}

export function getAnalysis() {
  return api.get('/analysis')
}

export function getTree() {
  return api.get('/tree')
}
