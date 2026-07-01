import axios from 'axios'

const api = axios.create({
  baseURL: '/api/prediction',
  timeout: 10000,
})

const studentApi = axios.create({
  baseURL: '/api/students',
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

export function getTrainOptions() {
  return api.get('/train/options')
}

export function train(config) {
  return api.post('/train', config, { timeout: 60000 })
}

export function getStudentHistory() {
  return studentApi.get('/history')
}
