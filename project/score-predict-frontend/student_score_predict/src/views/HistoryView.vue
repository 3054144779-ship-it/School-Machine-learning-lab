<script setup>
import { ref, computed, onMounted, onUnmounted } from 'vue'
import * as echarts from 'echarts'
import { getStudentHistory } from '@/api/index.js'

const loading = ref(false)
const error = ref('')
const students = ref([])
const page = ref(1)
const pageSize = 20
const totalPages = computed(() => Math.ceil(students.value.length / pageSize))
const pagedData = computed(() => {
  const start = (page.value - 1) * pageSize
  return students.value.slice(start, start + pageSize)
})

let distChart = null

onMounted(async () => {
  loading.value = true
  error.value = ''
  try {
    const res = await getStudentHistory()
    if (Array.isArray(res.data)) {
      students.value = res.data
    } else {
      error.value = '数据格式异常'
    }
  } catch (e) {
    error.value = '无法连接后端服务，请确认服务已启动'
  } finally {
    loading.value = false
    if (students.value.length > 0) {
      setTimeout(() => renderDistChart(), 100)
    }
  }
})

onUnmounted(() => { distChart?.dispose() })

const stats = computed(() => {
  if (students.value.length === 0) return null
  const avg = (arr, key) => {
    const vals = arr.map(s => s[key]).filter(v => v != null)
    return vals.length ? (vals.reduce((a, b) => a + b, 0) / vals.length).toFixed(1) : '-'
  }
  return {
    count: students.value.length,
    avgInteraction: avg(students.value, 'interaction'),
    avgComprehensive: avg(students.value, 'comprehensiveRegular'),
    avgOnlineTotal: avg(students.value, 'onlineTotal'),
  }
})

function renderDistChart() {
  const dom = document.getElementById('dist-chart')
  if (!dom) return
  distChart = echarts.init(dom)
  const scores = students.value.map(s => s.onlineTotal).filter(v => v != null)
  const bins = [0, 60, 70, 80, 90, 100]
  const counts = new Array(bins.length - 1).fill(0)
  const binLabels = ['<60', '60-70', '70-80', '80-90', '90-100']
  scores.forEach(s => {
    for (let i = 0; i < bins.length - 1; i++) {
      if (s >= bins[i] && s <= bins[i + 1]) { counts[i]++; break }
    }
  })
  distChart.setOption({
    tooltip: { trigger: 'axis' },
    xAxis: { type: 'category', data: binLabels, name: '线上总成绩区间' },
    yAxis: { type: 'value', name: '人数' },
    series: [{
      type: 'bar',
      data: counts,
      barWidth: '50%',
      itemStyle: {
        borderRadius: [6, 6, 0, 0],
        color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
          { offset: 0, color: '#3949ab' }, { offset: 1, color: '#5c6bc0' }
        ])
      }
    }]
  })
}

function goTo(pageNum) {
  if (pageNum >= 1 && pageNum <= totalPages.value) page.value = pageNum
}

const fieldMap = [
  { key: 'studentName', label: '学生姓名' },
  { key: 'interaction', label: '线下_互动' },
  { key: 'comprehensiveRegular', label: '综合_平时成绩' },
  { key: 'finalTotal', label: '期末总成绩' },
  { key: 'regularScore', label: '平时成绩' },
  { key: 'finalScore', label: '期末成绩' },
  { key: 'onlineTotal', label: '线上总成绩' },
]
</script>

<template>
  <div class="history-page">
    <div class="page-header">
      <h1>历史数据</h1>
      <p class="subtitle">共 {{ students.length }} 条学生成绩记录，来源：Score_dataset.xlsx (7个Sheet)</p>
    </div>

    <!-- Loading -->
    <div v-if="loading" class="state-card">
      <div class="spinner"></div>
      <p>正在加载历史数据...</p>
    </div>

    <!-- Error -->
    <div v-else-if="error" class="state-card error">
      <p>{{ error }}</p>
    </div>

    <!-- Empty -->
    <div v-else-if="students.length === 0" class="state-card">
      <p>暂无历史数据，请先运行 import_to_db.py 导入数据</p>
    </div>

    <!-- Content -->
    <div v-else>
      <!-- 统计卡片 -->
      <div class="stats-row">
        <div class="stat-card">
          <span class="stat-value">{{ stats?.count }}</span>
          <span class="stat-label">总记录数</span>
        </div>
        <div class="stat-card">
          <span class="stat-value">{{ stats?.avgInteraction }}</span>
          <span class="stat-label">线下互动均分</span>
        </div>
        <div class="stat-card">
          <span class="stat-value">{{ stats?.avgComprehensive }}</span>
          <span class="stat-label">综合平时成绩均分</span>
        </div>
        <div class="stat-card">
          <span class="stat-value">{{ stats?.avgOnlineTotal }}</span>
          <span class="stat-label">线上总成绩均分</span>
        </div>
      </div>

      <!-- 分布图 -->
      <div class="card chart-card">
        <h3>线上总成绩分布</h3>
        <div id="dist-chart" class="chart-box"></div>
      </div>

      <!-- 数据表格 -->
      <div class="card table-card">
        <h3>学生成绩明细</h3>
        <div class="table-wrap">
          <table>
            <thead>
              <tr>
                <th>#</th>
                <th v-for="f in fieldMap" :key="f.key">{{ f.label }}</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="(row, idx) in pagedData" :key="row.id">
                <td class="row-num">{{ (page - 1) * pageSize + idx + 1 }}</td>
                <td class="name-cell">{{ row.studentName || '-' }}</td>
                <td>{{ row.interaction != null ? row.interaction.toFixed(1) : '-' }}</td>
                <td>{{ row.comprehensiveRegular != null ? row.comprehensiveRegular.toFixed(1) : '-' }}</td>
                <td>{{ row.finalTotal != null ? row.finalTotal.toFixed(1) : '-' }}</td>
                <td>{{ row.regularScore != null ? row.regularScore.toFixed(1) : '-' }}</td>
                <td>{{ row.finalScore != null ? row.finalScore.toFixed(1) : '-' }}</td>
                <td :class="scoreClass(row.onlineTotal)">{{ row.onlineTotal != null ? row.onlineTotal.toFixed(1) : '-' }}</td>
              </tr>
            </tbody>
          </table>
        </div>

        <!-- 分页 -->
        <div class="pagination" v-if="totalPages > 1">
          <button :disabled="page <= 1" @click="goTo(page - 1)">上一页</button>
          <span class="page-info">{{ page }} / {{ totalPages }}</span>
          <button :disabled="page >= totalPages" @click="goTo(page + 1)">下一页</button>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  methods: {
    scoreClass(val) {
      if (val == null) return ''
      if (val >= 90) return 'score-excellent'
      if (val >= 80) return 'score-good'
      if (val >= 60) return 'score-medium'
      return 'score-fail'
    }
  }
}
</script>

<style scoped>
.history-page { max-width: 1200px; margin: 0 auto; }
.page-header { margin-bottom: 28px; }
.page-header h1 { font-size: 24px; font-weight: 600; color: #1a1a2e; }
.subtitle { color: #666; margin-top: 6px; font-size: 14px; }

.state-card {
  background: #fff; border-radius: 12px; padding: 60px 40px;
  text-align: center; box-shadow: 0 2px 12px rgba(0,0,0,0.06);
}
.state-card.error { border: 1px solid #ffcdd2; background: #fff5f5; color: #c62828; }

.spinner {
  width: 36px; height: 36px; border: 3px solid #e0e0e0;
  border-top-color: #1a237e; border-radius: 50%;
  animation: spin 0.8s linear infinite; margin: 0 auto 16px;
}
@keyframes spin { to { transform: rotate(360deg); } }

.stats-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 24px; }
@media (max-width: 768px) { .stats-row { grid-template-columns: repeat(2, 1fr); } }
.stat-card {
  background: #fff; border-radius: 12px; padding: 20px 24px;
  text-align: center; box-shadow: 0 2px 12px rgba(0,0,0,0.06);
}
.stat-value { display: block; font-size: 28px; font-weight: 700; color: #1a237e; }
.stat-label { display: block; font-size: 13px; color: #999; margin-top: 4px; }

.card {
  background: #fff; border-radius: 12px; padding: 28px;
  box-shadow: 0 2px 12px rgba(0,0,0,0.06); margin-bottom: 24px;
}
.card h3 {
  font-size: 17px; font-weight: 600; margin-bottom: 16px; color: #1a1a2e;
  padding-bottom: 12px; border-bottom: 2px solid #e8eaf6;
}
.chart-box { width: 100%; height: 320px; }

.table-wrap { overflow-x: auto; }
table { width: 100%; border-collapse: collapse; font-size: 14px; }
thead th {
  background: #f5f6fa; padding: 12px 10px; text-align: left;
  font-weight: 600; color: #555; border-bottom: 2px solid #e0e0e0;
  white-space: nowrap;
}
tbody td {
  padding: 10px 10px; border-bottom: 1px solid #f0f0f0; color: #333;
}
tbody tr:hover { background: #fafbff; }
.row-num { color: #999; font-size: 12px; width: 40px; }
.name-cell { font-weight: 500; }

.score-excellent { color: #2e7d32; font-weight: 600; }
.score-good { color: #1565c0; font-weight: 600; }
.score-medium { color: #e65100; font-weight: 600; }
.score-fail { color: #c62828; font-weight: 600; }

.pagination {
  display: flex; justify-content: center; align-items: center;
  gap: 16px; margin-top: 20px; padding-top: 16px; border-top: 1px solid #f0f0f0;
}
.pagination button {
  padding: 8px 20px; border: 1px solid #d0d0d0; border-radius: 6px;
  background: #fff; cursor: pointer; font-size: 14px; color: #555;
}
.pagination button:hover:not(:disabled) { background: #e8eaf6; border-color: #1a237e; color: #1a237e; }
.pagination button:disabled { opacity: 0.4; cursor: not-allowed; }
.page-info { font-size: 14px; color: #666; }
</style>
