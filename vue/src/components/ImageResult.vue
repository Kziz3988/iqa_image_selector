<template> 
  <div v-if="!isProcessing" class="result-page">
    <div class="result-header">
      <h2>图像优选结果</h2>
      <el-button type="primary" @click="downloadAllBest" v-if="bestImages.length">
        下载优选图像
      </el-button>
    </div>

    <div v-if="bestImages.length === 0">
      暂无最优结果
    </div>

    <div v-else class="image-grid">
      <div v-for="(bestImage, index) in bestImages" :key="'best-' + index" class="image-card">
        <img :src="getImageUrl(bestImage.file)" class="image" />
        <div class="info">
          <p>聚类: {{ bestImage.cluster }}</p>
          <p>分数: {{ bestImage.score.toFixed(3) }}</p>
          <p>IQA模型: {{ bestImage.model }}</p>
          <el-tag type="success">最优图像</el-tag>
        </div>
      </div>
    </div>

    <h2 v-if="otherImages.length !== 0">
      剩余图像
    </h2>
    <div v-if="otherImages.length !== 0" class="image-grid">
      <div v-for="(otherImage, index) in otherImages" :key="'other-' + index" class="image-card">
        <img :src="getImageUrl(otherImage.file)" class="image" />
        <div class="info">
          <p>聚类: {{ otherImage.cluster }}</p>
          <p>分数: {{ otherImage.score.toFixed(3) }}</p>
          <p>IQA模型: {{ otherImage.model }}</p>
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

  const bestImages = [];
  props.resultData.file_data.forEach(cluster => {
    const img = cluster.best_image;
    img.cluster = cluster.cluster;
    bestImages.push(img);
  });

  return bestImages;
});

const otherImages = computed(() => {
  if (!props.resultData) return [];

  const allOtherImages = [];
  props.resultData.file_data.forEach(cluster => {
    cluster.other_images.forEach(img => {
      img.cluster = cluster.cluster;
      allOtherImages.push(img);
    });
  });

  return allOtherImages;
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

.result-header {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 20px;
  margin-bottom: 20px;
}

.result-header h2 {
  margin: 0;
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