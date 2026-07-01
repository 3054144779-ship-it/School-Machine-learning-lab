<script setup>
import { ref, reactive, onMounted } from 'vue'
import { getFeatures, predict } from '@/api/index.js'

const loading = ref(false)
const submitting = ref(false)
const error = ref('')
const featureNames = ref([])
const result = ref(null)

// 滑块值，key 为特征名
const sliders = reactive({})

async function loadFeatures() {
  loading.value = true
  error.value = ''
  try {
    const res = await getFeatures()
    if (res.data.code === 200 && res.data.data) {
      featureNames.value = res.data.data.feature_names || []
      // 初始化每个滑块为中间值
      featureNames.value.forEach(name => {
        if (!(name in sliders)) {
          sliders[name] = 50
        }
      })
      // 清除已移除的特征
      Object.keys(sliders).forEach(key => {
        if (!featureNames.value.includes(key)) {
          delete sliders[key]
        }
      })
    } else {
      error.value = '加载特征信息失败'
    }
  } catch (e) {
    error.value = '无法连接后端服务，请确认服务已启动'
  } finally {
    loading.value = false
  }
}

onMounted(loadFeatures)

async function handleSubmit() {
  submitting.value = true
  error.value = ''
  result.value = null
  try {
    const values = featureNames.value.map(name => sliders[name])
    const res = await predict(values)
    if (res.data.code === 200 && res.data.data) {
      result.value = res.data.data
    } else {
      const msg = res.data.message || res.data.detail || '预测失败'
      if (msg.includes('特征数量不匹配') || msg.includes('维特征') || msg.includes('刷新页面')) {
        // 自动重载特征后重试一次
        await loadFeatures()
        const values2 = featureNames.value.map(name => sliders[name])
        const res2 = await predict(values2)
        if (res2.data.code === 200 && res2.data.data) {
          result.value = res2.data.data
        } else {
          error.value = res2.data.message || res2.data.detail || '预测失败'
        }
        return
      }
      error.value = msg
    }
  } catch (e) {
    const msg = e.response?.data?.detail || e.response?.data?.message || e.message
    if (msg && (msg.includes('特征数量不匹配') || msg.includes('维特征') || msg.includes('刷新页面'))) {
      await loadFeatures()
      const values2 = featureNames.value.map(name => sliders[name])
      const res2 = await predict(values2)
      if (res2.data.code === 200 && res2.data.data) {
        result.value = res2.data.data
      } else {
        error.value = res2.data?.message || res2.data?.detail || '预测失败'
      }
      return
    }
    error.value = '预测请求失败，请检查 Python 预测服务是否启动'
  } finally {
    submitting.value = false
  }
}

function getScoreClass(score) {
  if (score >= 90) return 'excellent'
  if (score >= 80) return 'good'
  if (score >= 60) return 'medium'
  return 'fail'
}

function getScoreLabel(score) {
  if (score >= 90) return '优'
  if (score >= 80) return '良'
  if (score >= 60) return '中'
  return '不及格'
}
</script>

<template>
  <div class="predict-page">
    <div class="page-header">
      <h1>个体成绩预测</h1>
      <p class="subtitle">调整学生各项指标，系统实时预测期末成绩</p>
    </div>

    <div v-if="loading" class="state-card">
      <div class="spinner"></div>
      <p>正在加载特征信息...</p>
    </div>

    <div v-else-if="error" class="state-card error">
      <p>{{ error }}</p>
      <button class="btn btn-outline" @click="() => { loading = true; error = ''; onMounted() }">重试</button>
    </div>

    <div v-else class="predict-content">
      <div class="card input-card">
        <h3>特征参数</h3>
        <div class="slider-group" v-for="name in featureNames" :key="name">
          <div class="slider-header">
            <label>{{ name }}</label>
            <span class="slider-value">{{ sliders[name] }}</span>
          </div>
          <input
            type="range"
            v-model.number="sliders[name]"
            min="0"
            max="100"
            step="1"
            class="slider"
          />
          <div class="slider-range">
            <span>0</span>
            <span>100</span>
          </div>
        </div>

        <button
          class="btn btn-primary"
          :disabled="submitting"
          @click="handleSubmit"
        >
          {{ submitting ? '预测中...' : '开始预测' }}
        </button>
      </div>

      <div class="card result-card" v-if="result">
        <h3>预测结果</h3>
        <div class="result-main">
          <div class="score-circle" :class="getScoreClass(result.predicted_score)">
            <span class="score-number">{{ result.predicted_score }}</span>
            <span class="score-unit">分</span>
          </div>
          <div class="result-detail">
            <div class="detail-row">
              <span class="detail-label">预测等级</span>
              <span class="detail-value tag" :class="getScoreClass(result.predicted_score)">
                {{ result.predicted_label || getScoreLabel(result.predicted_score) }}
              </span>
            </div>
            <div class="detail-row">
              <span class="detail-label">预测分数</span>
              <span class="detail-value">{{ result.predicted_score }} 分</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.predict-page {
  max-width: 900px;
  margin: 0 auto;
}

.page-header {
  margin-bottom: 28px;
}

.page-header h1 {
  font-size: 24px;
  font-weight: 600;
  color: #1a1a2e;
}

.subtitle {
  color: #666;
  margin-top: 6px;
  font-size: 14px;
}

.state-card {
  background: #fff;
  border-radius: 12px;
  padding: 60px 40px;
  text-align: center;
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.06);
}

.state-card.error {
  border: 1px solid #ffcdd2;
  background: #fff5f5;
  color: #c62828;
}

.spinner {
  width: 36px;
  height: 36px;
  border: 3px solid #e0e0e0;
  border-top-color: #1a237e;
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
  margin: 0 auto 16px;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

.predict-content {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 24px;
}

@media (max-width: 768px) {
  .predict-content {
    grid-template-columns: 1fr;
  }
}

.card {
  background: #fff;
  border-radius: 12px;
  padding: 28px;
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.06);
}

.card h3 {
  font-size: 17px;
  font-weight: 600;
  margin-bottom: 20px;
  color: #1a1a2e;
  padding-bottom: 12px;
  border-bottom: 2px solid #e8eaf6;
}

.slider-group {
  margin-bottom: 20px;
}

.slider-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 8px;
}

.slider-header label {
  font-size: 14px;
  font-weight: 500;
  color: #555;
}

.slider-value {
  font-size: 14px;
  font-weight: 600;
  color: #1a237e;
  background: #e8eaf6;
  padding: 2px 10px;
  border-radius: 10px;
}

.slider {
  width: 100%;
  height: 6px;
  -webkit-appearance: none;
  appearance: none;
  background: #e0e0e0;
  border-radius: 3px;
  outline: none;
  cursor: pointer;
}

.slider::-webkit-slider-thumb {
  -webkit-appearance: none;
  width: 20px;
  height: 20px;
  border-radius: 50%;
  background: #1a237e;
  cursor: pointer;
  box-shadow: 0 2px 6px rgba(26, 35, 126, 0.3);
}

.slider-range {
  display: flex;
  justify-content: space-between;
  font-size: 11px;
  color: #999;
  margin-top: 4px;
}

.btn {
  padding: 10px 24px;
  border: none;
  border-radius: 8px;
  font-size: 15px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s;
}

.btn-primary {
  width: 100%;
  margin-top: 8px;
  background: linear-gradient(135deg, #1a237e, #283593);
  color: #fff;
  padding: 14px;
  font-size: 16px;
}

.btn-primary:hover {
  box-shadow: 0 4px 16px rgba(26, 35, 126, 0.35);
  transform: translateY(-1px);
}

.btn-primary:disabled {
  opacity: 0.6;
  cursor: not-allowed;
  transform: none;
}

.btn-outline {
  margin-top: 16px;
  background: #fff;
  border: 1px solid #1a237e;
  color: #1a237e;
}

.result-main {
  display: flex;
  align-items: center;
  gap: 32px;
}

.score-circle {
  width: 120px;
  height: 120px;
  border-radius: 50%;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
}

.score-circle.excellent {
  background: linear-gradient(135deg, #e8f5e9, #c8e6c9);
  color: #2e7d32;
}

.score-circle.good {
  background: linear-gradient(135deg, #e3f2fd, #bbdefb);
  color: #1565c0;
}

.score-circle.medium {
  background: linear-gradient(135deg, #fff3e0, #ffe0b2);
  color: #e65100;
}

.score-circle.fail {
  background: linear-gradient(135deg, #ffebee, #ffcdd2);
  color: #c62828;
}

.score-number {
  font-size: 36px;
  font-weight: 700;
  line-height: 1;
}

.score-unit {
  font-size: 14px;
  margin-top: 2px;
}

.result-detail {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.detail-row {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.detail-label {
  font-size: 13px;
  color: #999;
}

.detail-value {
  font-size: 18px;
  font-weight: 600;
  color: #333;
}

.tag {
  display: inline-block;
  padding: 4px 14px;
  border-radius: 20px;
  font-size: 14px;
  align-self: flex-start;
}

.tag.excellent { background: #e8f5e9; color: #2e7d32; }
.tag.good { background: #e3f2fd; color: #1565c0; }
.tag.medium { background: #fff3e0; color: #e65100; }
.tag.fail { background: #ffebee; color: #c62828; }
</style>
