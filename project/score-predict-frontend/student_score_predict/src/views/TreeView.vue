<script setup>
import { ref, onMounted, onUnmounted } from 'vue'
import * as echarts from 'echarts'
import { getTree } from '@/api/index.js'

const loading = ref(false)
const error = ref('')
const treeData = ref(null)

let treeChart = null

onMounted(async () => {
  loading.value = true
  error.value = ''
  try {
    const res = await getTree()
    if (res.data.code === 200 && res.data.data) {
      treeData.value = res.data.data
    } else {
      error.value = res.data.message || '加载决策树数据失败'
    }
  } catch (e) {
    error.value = '无法连接后端服务，请确认服务已启动'
  } finally {
    loading.value = false
    if (treeData.value) {
      setTimeout(() => renderTree(), 100)
    }
  }
})

onUnmounted(() => {
  treeChart?.dispose()
})

function renderTree() {
  const dom = document.getElementById('tree-chart')
  if (!dom || !treeData.value) return
  treeChart = echarts.init(dom)
  treeChart.setOption({
    tooltip: { trigger: 'item', triggerOn: 'mousemove' },
    series: [{
      type: 'tree',
      data: [treeData.value],
      top: '5%',
      left: '8%',
      bottom: '5%',
      right: '15%',
      symbol: 'roundRect',
      symbolSize: 8,
      orient: 'LR',
      expandAndCollapse: false,
      label: {
        position: 'left',
        verticalAlign: 'middle',
        align: 'right',
        fontSize: 12,
      },
      leaves: {
        label: {
          position: 'right',
          verticalAlign: 'middle',
          align: 'left',
        },
      },
      lineStyle: {
        color: '#bbb',
        width: 1.5,
        curveness: 0.5,
      },
      itemStyle: {
        color: '#3949ab',
        borderColor: '#1a237e',
      },
      emphasis: {
        focus: 'descendant',
      },
    }],
  })
}

window.addEventListener('resize', () => {
  treeChart?.resize()
})
</script>

<template>
  <div class="tree-page">
    <div class="page-header">
      <h1>模型可视化</h1>
      <p class="subtitle">分类决策树结构 — 展示模型决策路径与分支规则</p>
    </div>

    <div v-if="loading" class="state-card">
      <div class="spinner"></div>
      <p>正在加载决策树数据...</p>
    </div>

    <div v-else-if="error" class="state-card error">
      <p>{{ error }}</p>
    </div>

    <div v-else class="card tree-card">
      <h3>决策树分支规则图</h3>
      <div id="tree-chart" class="tree-box"></div>
    </div>
  </div>
</template>

<style scoped>
.tree-page {
  max-width: 1200px;
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

.tree-box {
  width: 100%;
  height: 600px;
}
</style>
