<template>
  <div id="app">
    <div class="tab-container">
      <el-tabs v-model="activeTab" type="card">
        <el-tab-pane label="上传图像" name="upload"></el-tab-pane>
        <el-tab-pane label="优选结果" name="result" :disabled="!result"></el-tab-pane>
      </el-tabs>
    </div>

    <div class="tab-content">
      <ImageUpload v-if="activeTab === 'upload'" @upload-success="handleResult" />
      <ImageResult v-if="activeTab === 'result' && result" :resultData="result" />
      <div v-if="activeTab === 'result' && !result" class="no-result">
        暂无结果，请先上传图像
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, provide } from "vue";
import ImageUpload from './components/ImageUpload.vue';
import ImageResult from './components/ImageResult.vue';

const isProcessing = ref(false);
const result = ref(null);
const activeTab = ref("upload");

const handleResult = (data) => {
  result.value = data;
  activeTab.value = "result";
}

provide('isProcessing', isProcessing);
</script>

<style>
#app {
  font-family: Avenir, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  color: #2c3e50;
  margin-top: 0;
}

.tab-container {
  position: sticky;
  top: 0;
  z-index: 1000;
  background-color: #fff;
  padding: 10px 0;
  box-shadow: 0 2px 6px rgba(0,0,0,0.1);
}

.tab-content {
  padding: 20px;
}

.no-result {
  margin-top: 40px;
  font-size: 18px;
  color: #999;
}
</style>