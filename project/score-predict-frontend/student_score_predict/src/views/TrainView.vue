<script setup>
import { ref, onMounted, onUnmounted } from 'vue'
import * as echarts from 'echarts'
import { getTrainOptions, train } from '@/api/index.js'

// ========== 状态 ==========
const loading = ref(false)
const training = ref(false)
const error = ref('')
const successMsg = ref('')

// 可用选项
const availableFeatures = ref([])
const categoricalHints = ref([])
const totalSamples = ref(0)

// 用户选择的参数
const selectedFeatures = ref([])
const targetCol = ref('线上总成绩')
const testSize = ref(0.2)
const randomState = ref(42)
const maxDepth = ref(5)
const corrThreshold = ref(0.1)

// 训练结果
const trainResult = ref(null)

// 图表实例
let barChart = null
let heatmapChart = null

// 特征名映射（中文显示）
const featureLabelMap = {
  '线下_互动': '线下互动',
  '综合_平时成绩': '综合平时成绩',
  '期末总成绩': '期末总成绩',
  '平时成绩': '平时成绩',
  '期末成绩': '期末成绩',
  '参与度等级_中参与度': '参与度-中',
  '参与度等级_高参与度': '参与度-高',
}

function toLabel(name) {
  return featureLabelMap[name] || name
}

// ========== 初始化 ==========
onMounted(async () => {
  loading.value = true
  error.value = ''
  try {
    const res = await getTrainOptions()
    if (res.data.code === 200 && res.data.data) {
      const data = res.data.data
      availableFeatures.value = data.numeric_features || []
      categoricalHints.value = data.categorical_features || []
      totalSamples.value = data.total_samples || 0
      // 默认全选（排除明显是目标变量的列）
      const exclude = ['线上总成绩', '线上_平时成绩', '线上_期中测试', '线上_期末考试']
      selectedFeatures.value = availableFeatures.value.filter(f => !exclude.includes(f) && !f.startsWith('参与度等级'))
    } else {
      error.value = res.data.message || '加载训练选项失败'
    }
  } catch (e) {
    error.value = '无法连接后端服务，请确认服务已启动'
  } finally {
    loading.value = false
  }
})

onUnmounted(() => {
  barChart?.dispose()
  heatmapChart?.dispose()
})

// ========== 全选/取消全选 ==========
function toggleAll() {
  const exclude = ['线上总成绩', '线上_平时成绩', '线上_期中测试', '线上_期末考试']
  const selectable = availableFeatures.value.filter(f => !exclude.includes(f) && !f.startsWith('参与度等级'))
  if (selectedFeatures.value.length === selectable.length) {
    selectedFeatures.value = []
  } else {
    selectedFeatures.value = [...selectable]
  }
}

// ========== 训练 ==========
async function startTrain() {
  if (selectedFeatures.value.length === 0) {
    error.value = '请至少选择一个特征'
    return
  }
  training.value = true
  error.value = ''
  successMsg.value = ''
  trainResult.value = null

  try {
    const res = await train({
      target_col: targetCol.value,
      feature_names: selectedFeatures.value,
      test_size: testSize.value,
      random_state: randomState.value,
      max_depth: maxDepth.value,
      correlation_threshold: corrThreshold.value,
    })
    if (res.data.code === 200 && res.data.data) {
      trainResult.value = res.data.data
      successMsg.value = '模型训练完成！预测接口已使用新模型。'
      setTimeout(() => {
        renderBarChart()
        renderHeatmap()
      }, 150)
    } else {
      error.value = res.data.message || '训练失败'
    }
  } catch (e) {
    error.value = '训练请求失败: ' + (e.response?.data?.message || e.message)
  } finally {
    training.value = false
  }
}

// ========== 图表渲染 ==========
function renderBarChart() {
  const dom = document.getElementById('train-bar-chart')
  if (!dom || !trainResult.value?.feature_importance?.length) return
  barChart?.dispose()
  barChart = echarts.init(dom)

  const data = trainResult.value.feature_importance
  barChart.setOption({
    tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
    grid: { left: 80, right: 30, top: 20, bottom: 80 },
    xAxis: {
      type: 'category',
      data: data.map(d => toLabel(d.name)),
      axisLabel: { rotate: 30, fontSize: 11 },
    },
    yAxis: { type: 'value', name: '重要性' },
    series: [{
      type: 'bar',
      data: data.map((d, i) => ({
        value: d.value,
        itemStyle: {
          color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
            { offset: 0, color: '#3949ab' }, { offset: 1, color: '#5c6bc0' },
          ]),
          borderRadius: [4, 4, 0, 0],
        },
      })),
      barWidth: '40%',
    }],
  })
}

function renderHeatmap() {
  const dom = document.getElementById('train-heatmap-chart')
  if (!dom || !trainResult.value?.correlation_matrix?.length) return
  heatmapChart?.dispose()
  heatmapChart = echarts.init(dom)

  const labels = (trainResult.value.correlation_labels || trainResult.value.feature_names).map(toLabel)
  const matrix = trainResult.value.correlation_matrix
  const data = []
  for (let i = 0; i < matrix.length; i++) {
    for (let j = 0; j < matrix[i].length; j++) {
      data.push([j, i, matrix[i][j]])
    }
  }

  heatmapChart.setOption({
    tooltip: {
      formatter: (p) => `${labels[p.value[0]]} × ${labels[p.value[1]]}<br/>相关系数: ${p.value[2].toFixed(4)}`,
    },
    grid: { left: 120, right: 40, top: 20, bottom: 100 },
    xAxis: {
      type: 'category', data: labels,
      axisLabel: { rotate: 45, fontSize: 11 },
      position: 'bottom',
    },
    yAxis: {
      type: 'category', data: labels,
      axisLabel: { fontSize: 11 },
    },
    visualMap: {
      min: -1, max: 1, orient: 'horizontal', left: 'center', bottom: 0,
      inRange: { color: ['#1565c0', '#fff', '#c62828'] },
    },
    series: [{
      type: 'heatmap', data,
      label: { show: true, fontSize: 11 },
      emphasis: { itemStyle: { shadowBlur: 10, shadowColor: 'rgba(0,0,0,0.5)' } },
    }],
  })
}

window.addEventListener('resize', () => {
  barChart?.resize()
  heatmapChart?.resize()
})

// ========== 辅助 ==========
function formatPercent(v) {
  return v != null ? (v * 100).toFixed(2) + '%' : '--'
}
</script>

<template>
  <div class="train-page">
    <div class="page-header">
      <h1>模型训练</h1>
      <p class="subtitle">选择特征与参数，重新训练预测模型</p>
    </div>

    <!-- 加载 -->
    <div v-if="loading" class="state-card">
      <div class="spinner"></div>
      <p>正在加载可用特征...</p>
    </div>

    <!-- 错误 -->
    <div v-else-if="error && !trainResult" class="state-card error">
      <p>{{ error }}</p>
    </div>

    <div v-else class="train-layout">
      <!-- ====== 左侧：配置面板 ====== -->
      <div class="config-panel">
        <div class="card">
          <h3>特征选择 <span class="badge">{{ selectedFeatures.length }}/{{ availableFeatures.filter(f => !['线上总成绩', '线上_平时成绩', '线上_期中测试', '线上_期末考试'].includes(f) && !f.startsWith('参与度等级')).length }}</span></h3>
          <p class="hint">选择用于训练的特征列（共 {{ totalSamples }} 条样本）</p>

          <div class="feature-actions">
            <button class="btn-link" @click="toggleAll">全选 / 取消全选</button>
          </div>

          <div class="feature-list">
            <label
              v-for="f in availableFeatures"
              :key="f"
              class="feature-item"
              :class="{ excluded: ['线上总成绩', '线上_平时成绩', '线上_期中测试', '线上_期末考试'].includes(f) || f.startsWith('参与度等级') }"
            >
              <input
                type="checkbox"
                :value="f"
                v-model="selectedFeatures"
                :disabled="['线上总成绩', '线上_平时成绩', '线上_期中测试', '线上_期末考试'].includes(f) || f.startsWith('参与度等级')"
              />
              <span class="feature-name">{{ toLabel(f) }}</span>
              <span class="feature-orig" v-if="toLabel(f) !== f">({{ f }})</span>
            </label>
          </div>

          <!-- 独热编码提示 -->
          <div v-if="categoricalHints.length" class="onehot-hint">
            <p>💡 选择 <strong>线下_互动</strong> 后，训练时会自动生成参与度等级的独热编码特征：</p>
            <ul>
              <li v-for="h in categoricalHints" :key="h.source">
                {{ h.description }} → {{ h.onehot_labels.join(', ') }}
              </li>
            </ul>
          </div>
        </div>

        <div class="card">
          <h3>训练参数</h3>

          <div class="param-group">
            <label class="param-label">目标列</label>
            <select v-model="targetCol" class="param-select">
              <option value="线上总成绩">线上总成绩</option>
            </select>
          </div>

          <div class="param-group">
            <label class="param-label">
              测试集比例: <strong>{{ testSize }}</strong>
            </label>
            <input type="range" v-model.number="testSize" min="0.1" max="0.4" step="0.05" class="param-slider" />
            <span class="range-hint">0.1 ~ 0.4</span>
          </div>

          <div class="param-group">
            <label class="param-label">
              决策树最大深度: <strong>{{ maxDepth }}</strong>
            </label>
            <input type="range" v-model.number="maxDepth" min="2" max="15" step="1" class="param-slider" />
            <span class="range-hint">2 ~ 15</span>
          </div>

          <div class="param-group">
            <label class="param-label">
              随机种子: <strong>{{ randomState }}</strong>
            </label>
            <input type="number" v-model.number="randomState" min="0" max="999" class="param-input" />
          </div>

          <div class="param-group">
            <label class="param-label">
              相关性阈值: <strong>{{ corrThreshold }}</strong>
            </label>
            <input type="range" v-model.number="corrThreshold" min="0" max="0.5" step="0.05" class="param-slider" />
            <span class="range-hint">自动特征选择时的最低相关系数</span>
          </div>
        </div>

        <button
          class="btn-train"
          :disabled="training || selectedFeatures.length === 0"
          @click="startTrain"
        >
          <span v-if="training" class="spinner-small"></span>
          {{ training ? '训练中...' : '开始训练' }}
        </button>

        <div v-if="error" class="error-msg">{{ error }}</div>
        <div v-if="successMsg" class="success-msg">{{ successMsg }}</div>
      </div>

      <!-- ====== 右侧：结果面板 ====== -->
      <div class="result-panel">
        <div v-if="!trainResult && !training" class="state-card">
          <p class="placeholder-text">选择特征和参数后，点击「开始训练」</p>
        </div>

        <div v-if="training" class="state-card">
          <div class="spinner"></div>
          <p>模型正在训练中，请稍候...</p>
        </div>

        <template v-if="trainResult">
          <!-- 使用特征 -->
          <div class="card">
            <h3>训练配置</h3>
            <div class="config-summary">
              <p>使用特征 ({{ trainResult.feature_names.length }}): <code>{{ trainResult.feature_names.join(', ') }}</code></p>
              <p>目标列: <code>{{ trainResult.target_col }}</code></p>
              <p>分类标签: <code>{{ trainResult.class_labels.join(' → ') }}</code></p>
            </div>
          </div>

          <!-- 评估指标 -->
          <div class="metrics-row">
            <div class="card metric-card">
              <h3>多元线性回归</h3>
              <div class="metric-grid">
                <div class="metric-item">
                  <span class="metric-val">{{ trainResult.metrics.linear_regression.r2 }}</span>
                  <span class="metric-label">R²</span>
                </div>
                <div class="metric-item">
                  <span class="metric-val">{{ trainResult.metrics.linear_regression.mae }}</span>
                  <span class="metric-label">MAE</span>
                </div>
                <div class="metric-item">
                  <span class="metric-val">{{ trainResult.metrics.linear_regression.rmse }}</span>
                  <span class="metric-label">RMSE</span>
                </div>
              </div>

              <!-- 特征权重 -->
              <div v-if="trainResult.metrics.linear_regression.feature_weights?.length" class="weights-table">
                <h4>特征权重</h4>
                <table>
                  <thead><tr><th>特征</th><th>权重</th><th>方向</th></tr></thead>
                  <tbody>
                    <tr v-for="w in trainResult.metrics.linear_regression.feature_weights" :key="w.name">
                      <td>{{ toLabel(w.name) }}</td>
                      <td :class="w.weight > 0 ? 'positive' : 'negative'">{{ w.weight }}</td>
                      <td>{{ w.direction }}</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>

            <div class="card metric-card">
              <h3>分类决策树</h3>
              <div class="metric-grid">
                <div class="metric-item">
                  <span class="metric-val">{{ formatPercent(trainResult.metrics.decision_tree.accuracy) }}</span>
                  <span class="metric-label">准确率</span>
                </div>
                <div class="metric-item">
                  <span class="metric-val">{{ trainResult.metrics.decision_tree.precision }}</span>
                  <span class="metric-label">精确率</span>
                </div>
                <div class="metric-item">
                  <span class="metric-val">{{ trainResult.metrics.decision_tree.recall }}</span>
                  <span class="metric-label">召回率</span>
                </div>
              </div>

              <!-- 混淆矩阵 -->
              <div v-if="trainResult.metrics.decision_tree.confusion_matrix" class="confusion-matrix">
                <h4>混淆矩阵</h4>
                <table>
                  <thead>
                    <tr><th></th><th v-for="l in trainResult.class_labels" :key="l">预测{{ l }}</th></tr>
                  </thead>
                  <tbody>
                    <tr v-for="(row, i) in trainResult.metrics.decision_tree.confusion_matrix" :key="i">
                      <td><strong>真实{{ trainResult.class_labels[i] }}</strong></td>
                      <td v-for="(v, j) in row" :key="j" :class="{ diag: i === j, off: i !== j && v > 0 }">{{ v }}</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          <!-- 特征重要性 -->
          <div class="card chart-card" v-if="trainResult.feature_importance?.length">
            <h3>特征重要性（决策树）</h3>
            <div id="train-bar-chart" class="chart-box"></div>
          </div>

          <!-- 相关性热力图 -->
          <div class="card chart-card" v-if="trainResult.correlation_matrix?.length">
            <h3>特征相关性热力图</h3>
            <div id="train-heatmap-chart" class="chart-box"></div>
          </div>
        </template>
      </div>
    </div>
  </div>
</template>

<style scoped>
.train-page {
  max-width: 1300px;
  margin: 0 auto;
}

.page-header {
  margin-bottom: 24px;
}
.page-header h1 {
  font-size: 24px;
  font-weight: 600;
  color: #1a1a2e;
}
.subtitle {
  color: #666;
  margin-top: 4px;
  font-size: 14px;
}

/* ---- 布局 ---- */
.train-layout {
  display: flex;
  gap: 24px;
  align-items: flex-start;
}
.config-panel {
  width: 340px;
  flex-shrink: 0;
  display: flex;
  flex-direction: column;
  gap: 16px;
  position: sticky;
  top: 16px;
}
.result-panel {
  flex: 1;
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: 16px;
}

/* ---- 卡片 ---- */
.card {
  background: #fff;
  border-radius: 12px;
  padding: 22px;
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.06);
}
.card h3 {
  font-size: 16px;
  font-weight: 600;
  margin-bottom: 14px;
  color: #1a1a2e;
  display: flex;
  align-items: center;
  gap: 8px;
}
.card h4 {
  font-size: 14px;
  font-weight: 600;
  color: #555;
  margin: 14px 0 8px;
}
.badge {
  font-size: 12px;
  background: #e8eaf6;
  color: #3949ab;
  padding: 2px 8px;
  border-radius: 10px;
  font-weight: 500;
}
.hint {
  font-size: 12px;
  color: #999;
  margin-bottom: 10px;
}

/* ---- 特征列表 ---- */
.feature-actions {
  margin-bottom: 8px;
}
.btn-link {
  background: none;
  border: none;
  color: #3949ab;
  cursor: pointer;
  font-size: 13px;
  padding: 0;
}
.btn-link:hover { text-decoration: underline; }

.feature-list {
  max-height: 280px;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 2px;
}
.feature-item {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 10px;
  border-radius: 6px;
  cursor: pointer;
  transition: background 0.15s;
  font-size: 13px;
}
.feature-item:hover { background: #f5f5f5; }
.feature-item.excluded { opacity: 0.45; cursor: not-allowed; }
.feature-name { flex: 1; }
.feature-orig { color: #aaa; font-size: 11px; }

.onehot-hint {
  margin-top: 12px;
  background: #fff8e1;
  border-radius: 8px;
  padding: 12px;
  font-size: 12px;
  color: #6d4c00;
  line-height: 1.6;
}
.onehot-hint ul {
  margin: 4px 0 0 16px;
}

/* ---- 参数 ---- */
.param-group {
  margin-bottom: 14px;
}
.param-label {
  display: block;
  font-size: 13px;
  color: #555;
  margin-bottom: 6px;
}
.param-slider {
  width: 100%;
  accent-color: #3949ab;
}
.param-select, .param-input {
  width: 100%;
  padding: 8px 10px;
  border: 1px solid #ddd;
  border-radius: 6px;
  font-size: 14px;
}
.range-hint {
  font-size: 11px;
  color: #aaa;
}

/* ---- 按钮 ---- */
.btn-train {
  width: 100%;
  padding: 12px;
  background: linear-gradient(135deg, #1a237e, #3949ab);
  color: #fff;
  border: none;
  border-radius: 8px;
  font-size: 15px;
  font-weight: 600;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  transition: opacity 0.2s;
}
.btn-train:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}
.btn-train:hover:not(:disabled) {
  background: linear-gradient(135deg, #0d1557, #283593);
}

.error-msg {
  color: #c62828;
  font-size: 13px;
  text-align: center;
  padding: 8px;
  background: #fff5f5;
  border-radius: 6px;
}
.success-msg {
  color: #2e7d32;
  font-size: 13px;
  text-align: center;
  padding: 8px;
  background: #e8f5e9;
  border-radius: 6px;
}

/* ---- 结果 ---- */
.config-summary {
  font-size: 13px;
  line-height: 1.8;
}
.config-summary code {
  background: #f0f0f0;
  padding: 1px 6px;
  border-radius: 3px;
  font-size: 12px;
}

.metrics-row {
  display: flex;
  gap: 16px;
}
.metric-card {
  flex: 1;
}
.metric-grid {
  display: flex;
  gap: 12px;
  margin-bottom: 8px;
}
.metric-item {
  flex: 1;
  text-align: center;
  background: #f8f9ff;
  border-radius: 8px;
  padding: 14px 8px;
}
.metric-val {
  display: block;
  font-size: 22px;
  font-weight: 700;
  color: #1a237e;
}
.metric-label {
  font-size: 12px;
  color: #888;
  margin-top: 4px;
}

/* 权重表格 */
.weights-table table, .confusion-matrix table {
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
}
.weights-table th, .weights-table td,
.confusion-matrix th, .confusion-matrix td {
  padding: 6px 8px;
  border-bottom: 1px solid #eee;
  text-align: center;
}
.weights-table th { color: #888; font-weight: 500; }
.positive { color: #2e7d32; }
.negative { color: #c62828; }
.diag { background: #e8f5e9; font-weight: 600; }
.off { background: #fff5f5; }

/* 图表 */
.chart-card { padding-bottom: 10px; }
.chart-box {
  width: 100%;
  height: 380px;
}

/* 状态卡片 */
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
.placeholder-text { color: #aaa; font-size: 15px; }

.spinner {
  width: 36px;
  height: 36px;
  border: 3px solid #e0e0e0;
  border-top-color: #1a237e;
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
  margin: 0 auto 16px;
}
.spinner-small {
  width: 18px;
  height: 18px;
  border: 2px solid rgba(255,255,255,0.4);
  border-top-color: #fff;
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }
</style>
