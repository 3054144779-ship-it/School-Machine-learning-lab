import { createRouter, createWebHistory } from 'vue-router'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    { path: '/', redirect: '/predict' },
    { path: '/predict', name: 'predict', component: () => import('@/views/PredictView.vue') },
    { path: '/analysis', name: 'analysis', component: () => import('@/views/AnalysisView.vue') },
    { path: '/tree', name: 'tree', component: () => import('@/views/TreeView.vue') },
    { path: '/train', name: 'train', component: () => import('@/views/TrainView.vue') },
    { path: '/history', name: 'history', component: () => import('@/views/HistoryView.vue') },
  ],
})

export default router
