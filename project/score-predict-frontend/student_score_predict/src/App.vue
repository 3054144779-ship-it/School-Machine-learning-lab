<script setup>
import { useRoute } from 'vue-router'
import { computed } from 'vue'

const route = useRoute()
const activeMenu = computed(() => route.path)

const menuItems = [
  { path: '/predict', label: '个体预测', icon: '📊' },
  { path: '/analysis', label: '总体分析', icon: '📈' },
  { path: '/train', label: '模型训练', icon: '⚙️' },
  { path: '/tree', label: '模型可视化', icon: '🌳' },
  { path: '/history', label: '历史数据', icon: '📋' },
]
</script>

<template>
  <div class="app-container">
    <aside class="sidebar">
      <div class="sidebar-header">
        <h2>成绩预测系统</h2>
      </div>
      <nav class="nav-menu">
        <router-link
          v-for="item in menuItems"
          :key="item.path"
          :to="item.path"
          class="nav-item"
          :class="{ active: activeMenu === item.path }"
        >
          <span class="nav-icon">{{ item.icon }}</span>
          <span class="nav-label">{{ item.label }}</span>
        </router-link>
      </nav>
    </aside>
    <main class="main-content">
      <router-view />
    </main>
  </div>
</template>

<style>
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
  background: #f0f2f5;
  color: #333;
}

.app-container {
  display: flex;
  min-height: 100vh;
}

.sidebar {
  width: 220px;
  background: linear-gradient(180deg, #1a237e 0%, #283593 100%);
  color: #fff;
  display: flex;
  flex-direction: column;
  flex-shrink: 0;
}

.sidebar-header {
  padding: 24px 20px;
  border-bottom: 1px solid rgba(255, 255, 255, 0.12);
}

.sidebar-header h2 {
  font-size: 18px;
  font-weight: 600;
  letter-spacing: 1px;
}

.nav-menu {
  padding: 12px 0;
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.nav-item {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 14px 20px;
  color: rgba(255, 255, 255, 0.7);
  text-decoration: none;
  transition: all 0.2s ease;
  border-left: 3px solid transparent;
  font-size: 15px;
}

.nav-item:hover {
  color: #fff;
  background: rgba(255, 255, 255, 0.08);
}

.nav-item.active {
  color: #fff;
  background: rgba(255, 255, 255, 0.15);
  border-left-color: #64b5f6;
}

.nav-icon {
  font-size: 18px;
  width: 24px;
  text-align: center;
}

.main-content {
  flex: 1;
  padding: 32px;
  overflow-y: auto;
}
</style>
