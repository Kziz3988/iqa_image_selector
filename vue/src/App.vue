<template>
  <div id="app">
    <nav class="navbar">
      <div class="nav-container">
        <div class="nav-logo">
          <span>相似图像优选系统</span>
        </div>
        <div class="nav-menu">
          <div 
            class="nav-item"
            :class="{ active: activeTab === 'upload' }"
            @click="switchTab('upload')"
          >
            上传图像
          </div>
          <div 
            class="nav-item"
            :class="{ active: activeTab === 'result', disabled: !result && activeTab !== 'upload' }"
            @click="switchTab('result')"
          >
            优选结果
          </div>
        </div>
      </div>
    </nav>

    <div class="content-wrapper">
      <div class="tab-content">
        <ImageUpload v-if="activeTab === 'upload'" @upload-success="handleResult" />
        <ImageResult v-if="activeTab === 'result' && result" :resultData="result" />
        <div v-if="activeTab === 'result' && !result" class="no-result">
          <el-empty description="暂无结果，请先上传图像" :image-size="120" />
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, provide } from "vue";
import ImageUpload from './components/ImageUpload.vue';
import ImageResult from './components/ImageResult.vue';
import { ElEmpty } from 'element-plus';

const isProcessing = ref(false);
const result = ref(null);
const activeTab = ref("upload");

const switchTab = (tab) => {
  if (tab === 'result' && !result.value) {
    return;
  }
  activeTab.value = tab;
};

const handleResult = (data) => {
  result.value = data;
  activeTab.value = "result";
};

provide('isProcessing', isProcessing);
</script>

<style scoped>
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

#app {
  font-family: 'Avenir', 'Helvetica', 'Arial', sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  color: #2c3e50;
  background-color: #f5f7fa;
  min-height: 100vh;
}

.navbar {
  position: sticky;
  top: 0;
  z-index: 1000;
  background: linear-gradient(135deg, #1565C0 0%, #1e88e5 100%);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
  backdrop-filter: blur(0px);
  transition: all 0.3s ease;
}

.nav-container {
  max-width: 1400px;
  margin: 0 auto;
  padding: 0 30px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  height: 64px;
}

.nav-logo {
  font-size: 20px;
  font-weight: 600;
  color: white;
  letter-spacing: 1px;
  cursor: default;
  background: rgba(255, 255, 255, 0.1);
  padding: 6px 16px;
  border-radius: 32px;
  transition: all 0.3s;
}

.nav-logo span {
  background: linear-gradient(135deg, #fff, #e3f2fd);
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
}

.nav-menu {
  display: flex;
  gap: 8px;
  background: rgba(255, 255, 255, 0.1);
  padding: 4px;
  border-radius: 48px;
  backdrop-filter: blur(10px);
}

.nav-item {
  padding: 8px 24px;
  font-size: 16px;
  font-weight: 500;
  color: rgba(255, 255, 255, 0.85);
  cursor: pointer;
  border-radius: 40px;
  transition: all 0.25s ease;
  letter-spacing: 0.5px;
  position: relative;
}

.nav-item:hover:not(.disabled) {
  background: rgba(255, 255, 255, 0.2);
  color: white;
  transform: translateY(-1px);
}

.nav-item.active {
  background: white;
  color: #1565C0;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.nav-item.disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.content-wrapper {
  max-width: 1400px;
  margin: 0 auto;
  padding: 32px 30px;
}

.tab-content {
  animation: fadeIn 0.3s ease;
}

.no-result {
  margin-top: 60px;
  display: flex;
  justify-content: center;
  align-items: center;
  min-height: 400px;
}

@keyframes fadeIn {
  from {
    opacity: 0;
    transform: translateY(10px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

@media (max-width: 768px) {
  .nav-container {
    padding: 0 16px;
    height: 56px;
  }
  
  .nav-logo {
    font-size: 16px;
    padding: 4px 12px;
  }
  
  .nav-item {
    padding: 6px 16px;
    font-size: 14px;
  }
  
  .content-wrapper {
    padding: 20px 16px;
  }
}

@media (max-width: 480px) {
  .nav-item {
    padding: 4px 12px;
    font-size: 13px;
  }
  
  .nav-logo {
    font-size: 14px;
  }
}
</style>