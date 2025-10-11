import axios from 'axios'

const DB_API_BASE_URL = process.env.NEXT_PUBLIC_DB_API_URL || 'http://localhost:8001'

const dbApi = axios.create({
  baseURL: DB_API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Request interceptor
dbApi.interceptors.request.use(
  (config) => {
    console.log(`DB API Request: ${config.method?.toUpperCase()} ${config.url}`)
    return config
  },
  (error) => {
    console.error('DB API Request Error:', error)
    return Promise.reject(error)
  }
)

// Response interceptor
dbApi.interceptors.response.use(
  (response) => {
    console.log(`DB API Response: ${response.status} ${response.config.url}`)
    return response
  },
  (error) => {
    console.error('DB API Response Error:', error.response?.data || error.message)
    return Promise.reject(error)
  }
)

export default dbApi
