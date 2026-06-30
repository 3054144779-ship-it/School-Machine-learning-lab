<script setup>
import { ref, onMounted, onUnmounted } from 'vue'
import * as echarts from 'echarts'
import { getAnalysis } from '@/api/index.js'

const loading = ref(false)
const error = ref('')
const featureImportance = ref([])
const correlationMatrix = ref([])
const featureLabels = ref([])

let barChart = null
let heatmapChart = null

onMounted(async () => {
  loading.value = true
  error.value = ''
  try {
    const res = await getAnalysis()
    if (res.data.code === 200 && res.data.data) {
      const data = res.data.data
      featureImportance.value = data.feature_importance || []
      correlationMatrix.value = data.correlation_matrix || []
      if (featureImportance.value.length > 0) {
        featureLabels.value = featureImportance.value.map(d => d.name)
      }
    } else {
      error.value = res.data.message || '加载分析数据失败'
    }
  } catch (e) {
    error.value = '无法连接后端服务，请确认服务已启动'
  } finally {
    loading.value = false
    if (!error.value) {
      setTimeout(() => {
        renderBarChart()
        renderHeatmap()
      }, 100)
    }
  }
})

onUnmounted(() => {
  barChart?.dispose()
  heatmapChart?.dispose()
})

function renderBarChart() {
  const dom = document.getElementById('bar-chart')
  if (!dom || featureImportance.value.length === 0) return
  barChart = echarts.init(dom)
  barChart.setOption({
    tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
    grid: { left: 60, right: 30, top: 20, bottom: 60 },
    xAxis: {
      type: 'category',
      data: featureImportance.value.map(d => d.name),
      axisLabel: { rotate: 30, fontSize: 12 },
    },
    yAxis: { type: 'value', name: '重要性' },
    series: [{
      type: 'bar',
      data: featureImportance.value.map((d, i) => ({
        value: d.value,
        itemStyle: {
          color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
            { offset: 0, color: '#3949ab' },
            { offset: 1, color: '#5c6bc0' },
          ]),
          borderRadius: [4, 4, 0, 0],
        },
      })),
      barWidth: '40%',
    }],
  })
}

function renderHeatmap() {
  const dom = document.getElementById('heatmap-chart')
  if (!dom || correlationMatrix.value.length === 0) return
  heatmapChart = echarts.init(dom)

  const labels = featureLabels.value
  const data = []
  for (let i = 0; i < correlationMatrix.value.length; i++) {
    for (let j = 0; j < correlationMatrix.value[i].length; j++) {
      data.push([j, i, correlationMatrix.value[i][j]])
    }
  }

  heatmapChart.setOption({
    tooltip: {
      formatter: (p) => `${labels[p.value[0]]} × ${labels[p.value[1]]}<br/>相关系数: ${p.value[2].toFixed(4)}`,
    },
    grid: { left: 120, right: 40, top: 20, bottom: 100 },
    xAxis: {
      type: 'category',
      data: labels,
      axisLabel: { rotate: 45, fontSize: 11 },
      position: 'bottom',
    },
    yAxis: {
      type: 'category',
      data: labels,
      axisLabel: { fontSize: 11 },
    },
    visualMap: {
      min: -1,
      max: 1,
      orient: 'horizontal',
      left: 'center',
      bottom: 0,
      inRange: { color: ['#1565c0', '#fff', '#c62828'] },
    },
    series: [{
      type: 'heatmap',
      data,
      label: { show: true, fontSize: 11 },
      emphasis: { itemStyle: { shadowBlur: 10, shadowColor: 'rgba(0,0,0,0.5)' } },
    }],
  })
}

window.addEventListener('resize', () => {
  barChart?.resize()
  heatmapChart?.resize()
})
</script>

<template>
  <div class="analysis-page">
    <div class="page-header">
      <h1>总体分析</h1>
      <p class="subtitle">特征重要性分析 &amp; 特征相关性热力图</p>
    </div>

    <div v-if="loading" class="state-card">
      <div class="spinner"></div>
      <p>正在加载分析数据...</p>
    </div>

    <div v-else-if="error" class="state-card error">
      <p>{{ error }}</p>
    </div>

    <div v-else class="analysis-content">
      <div class="card chart-card">
        <h3>特征重要性（决策树）</h3>
        <div v-if="featureImportance.length === 0" class="empty-hint">暂无数据</div>
        <div id="bar-chart" class="chart-box" v-show="featureImportance.length > 0"></div>
      </div>

      <div class="card chart-card">
        <h3>特征相关性热力图</h3>
        <div v-if="correlationMatrix.length === 0" class="empty-hint">暂无数据，请先运行 main.py 生成清洗数据</div>
        <div id="heatmap-chart" class="chart-box" v-show="correlationMatrix.length > 0"></div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.analysis-page {
  max-width: 1100px;
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

.analysis-content {
  display: flex;
  flex-direction: column;
  gap: 24px;
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
  margin-bottom: 16px;
  color: #1a1a2e;
  padding-bottom: 12px;
  border-bottom: 2px solid #e8eaf6;
}

.chart-box {
  width: 100%;
  height: 400px;
}

.empty-hint {
  text-align: center;
  color: #999;
  padding: 60px 0;
  font-size: 14px;
}
</style>
