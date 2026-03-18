<template>
  <div v-if="!isProcessing" class="result-page">
    <h2>图像优选结果</h2>

    <div v-if="bestImages.length === 0">
      暂无最优结果
    </div>

    <div v-else class="image-grid">
      <div v-for="(file, index) in bestImages" :key="index" class="image-card">
        <img :src="getImageUrl(file)" class="image" />
        <div class="info">
          <p>聚类: {{ resultData.labels[resultData.all_files.indexOf(file)] }}</p>
          <p>分数: {{ resultData.scores[resultData.all_files.indexOf(file)].toFixed(3) }}</p>
          <el-tag type="success">最优图像</el-tag>
        </div>
      </div>
    </div>

    <el-button type="primary" @click="downloadAllBest" style="margin-top: 20px;">
      下载优选图像
    </el-button>

    <h2 v-if="otherImages.length !== 0">
      剩余图像
    </h2>
    <div v-if="otherImages.length !== 0" class="image-grid">
      <div v-for="(file, index) in otherImages" :key="index" class="image-card">
        <img :src="getImageUrl(file)" class="image" />
        <div class="info">
          <p>聚类: {{ resultData.labels[resultData.all_files.indexOf(file)] }}</p>
          <p>分数: {{ resultData.scores[resultData.all_files.indexOf(file)].toFixed(3) }}</p>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { defineProps, computed, inject } from "vue";
import JSZip from "jszip";
import { saveAs } from "file-saver";
const isProcessing = inject('isProcessing')

const props = defineProps({
  resultData: Object
});

const bestImages = computed(() => {
  if (!props.resultData) return [];

  const best = Object.values(props.resultData.best_in_cluster);
  return props.resultData.all_files
    .filter(file => best.includes(file))
    .sort((a, b) => {
      const idxA = props.resultData.all_files.indexOf(a);
      const idxB = props.resultData.all_files.indexOf(b);
      const labelA = props.resultData.labels[idxA];
      const labelB = props.resultData.labels[idxB];
      if (labelA !== labelB) return labelA - labelB;
      const scoreA = props.resultData.scores[idxA];
      const scoreB = props.resultData.scores[idxB];
      return scoreB - scoreA;
    });
});

const otherImages = computed(() => {
  if (!props.resultData) return [];

  const best = Object.values(props.resultData.best_in_cluster);
  return props.resultData.all_files
    .filter(file => !best.includes(file))
    .sort((a, b) => {
      const idxA = props.resultData.all_files.indexOf(a);
      const idxB = props.resultData.all_files.indexOf(b);
      const labelA = props.resultData.labels[idxA];
      const labelB = props.resultData.labels[idxB];
      if (labelA !== labelB) return labelA - labelB;
      const scoreA = props.resultData.scores[idxA];
      const scoreB = props.resultData.scores[idxB];
      return scoreB - scoreA;
    });
});

const getImageUrl = (file) => {
  const raw = props.resultData.fileMap[file];
  if (!raw) return "";
  return URL.createObjectURL(raw);
};

const downloadAllBest = async () => {
  if (!bestImages.value.length) return;
  const zip = new JSZip();
  bestImages.value.forEach((file) => {
    const blob = props.resultData.fileMap[file];
    if (blob) {
      zip.file(file, blob);
    }
  });
  const content = await zip.generateAsync({ type: "blob" });
  saveAs(content, "best_images.zip");
};
</script>

<style scoped>
.result-page {
  max-width: 1200px;
  margin: 0 auto;
  padding: 40px auto;
  text-align: center;
}

.image-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, 200px);
  gap: 30px;
  justify-content: center;
}

.image {
  width: 100%;
  height: 160px;
  object-fit: cover;
}

.image-card {
  width: 200px;
  border: 1px solid #ddd;
  padding: 10px;
  border-radius: 8px;
  transition: 0.25s;
}

.image-card:hover {
  transform: translateY(-4px);
  box-shadow: 0 8px 18px rgba(0, 0, 0, 0.15);
}

.info {
  margin-top: 8px;
}
</style>