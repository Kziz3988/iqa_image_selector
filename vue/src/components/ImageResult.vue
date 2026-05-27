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
      <div v-for="(bestImage, index) in bestImages" :key="'best-' + index" class="image-card" @click="handlePreview(bestImage.uuid)">
        <img :src="getImageUrl(bestImage.uuid)" class="image" @error="handleImageError(bestImage.uuid)" />
        <div class="info">
          <p>聚类: {{ bestImage.cluster }}</p>
          <p>分数: {{ bestImage.score.toFixed(3) }}</p>
          <p>IQA模型: {{ bestImage.model }}</p>
          <el-tag type="success">最优图像</el-tag>
        </div>
      </div>
    </div>

    <!-- 虚线分隔符 -->
    <div v-if="otherImages.length !== 0" class="divider"></div>

    <h2 v-if="otherImages.length !== 0">
      剩余图像
    </h2>
    <div v-if="otherImages.length !== 0" class="image-grid">
      <div v-for="(otherImage, index) in otherImages" :key="'other-' + index" class="image-card" @click="handlePreview(otherImage.uuid)">
        <img :src="getImageUrl(otherImage.uuid)" class="image" @error="handleImageError(otherImage.uuid)" />
        <div class="info">
          <p>聚类: {{ otherImage.cluster }}</p>
          <p>分数: {{ otherImage.score.toFixed(3) }}</p>
          <p>IQA模型: {{ otherImage.model }}</p>
        </div>
      </div>
    </div>

    <el-dialog v-model="dialogVisible">
      <img
        :style="{ width: '100%' }"
        :src="previewImageUrl"
        @load="onPreviewImageLoad"
      />
    </el-dialog>
  </div>
</template>

<script setup>
import { defineProps, computed, inject, ref } from "vue";
import JSZip from "jszip";
import { saveAs } from "file-saver";

const isProcessing = inject('isProcessing')

const props = defineProps({
  resultData: Object
});

// 预览相关状态
const dialogVisible = ref(false);
const previewImageUrl = ref("");
const currentPreviewUuid = ref(null);

const bestImages = computed(() => {
  if (!props.resultData) return [];

  const bestImages = [];
  props.resultData.file_data.forEach(cluster => {
    const img = { ...cluster.best_image };
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
      const newImg = { ...img };
      newImg.cluster = cluster.cluster;
      allOtherImages.push(newImg);
    });
  });

  return allOtherImages;
});

const getImageUrl = (uuid) => {
  if (!props.resultData || !props.resultData.fileStorage) {
    console.warn('No fileStorage for getImageUrl, uuid:', uuid)
    return ""
  }
  const fileInfo = props.resultData.fileStorage.get(uuid)
  if (!fileInfo || !fileInfo.blob) {
    console.warn('No blob found for uuid:', uuid, 'fileInfo:', fileInfo)
    return ""
  }
  const url = URL.createObjectURL(fileInfo.blob)
  console.log('Created URL for uuid:', uuid, 'url:', url)
  return url
};

const handleImageError = (uuid) => {
  console.error('Failed to load image for uuid:', uuid)
}

// 预览处理函数
const handlePreview = (uuid) => {
  if (currentPreviewUuid.value === uuid && dialogVisible.value) {
    // 如果点击的是同一张图片且弹窗已打开，不做处理
    return;
  }
  
  // 清理之前的 object URL
  if (previewImageUrl.value && previewImageUrl.value.startsWith('blob:')) {
    URL.revokeObjectURL(previewImageUrl.value);
  }
  
  currentPreviewUuid.value = uuid;
  const url = getImageUrl(uuid);
  if (url) {
    previewImageUrl.value = url;
    dialogVisible.value = true;
  } else {
    console.error('无法获取图片URL:', uuid);
  }
};

// 预览图片加载完成后的处理（可选，用于调试）
const onPreviewImageLoad = () => {
  console.log('Preview image loaded for uuid:', currentPreviewUuid.value);
};

const downloadAllBest = async () => {
  if (!bestImages.value.length) return;
  const zip = new JSZip();
  
  bestImages.value.forEach((img) => {
    const fileInfo = props.resultData.fileStorage.get(img.uuid)
    if (fileInfo && fileInfo.blob) {
      zip.file(fileInfo.originalName, fileInfo.blob)
    } else {
      console.warn("missing file:", img.uuid)
    }
  });

  const content = await zip.generateAsync({ type: "blob" })
  saveAs(content, "best_images.zip")
}
</script>

<style scoped>
/* 样式保持不变 */
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
  cursor: pointer;
}

.image-card:hover {
  transform: translateY(-4px);
  box-shadow: 0 8px 18px rgba(0, 0, 0, 0.15);
}

.info {
  margin-top: 8px;
}

.divider {
  width: 80%;
  height: 2px;
  background: repeating-linear-gradient(
    to right,
    #dcdfe6,
    #dcdfe6 12px,
    transparent 12px,
    transparent 24px
  );
  margin: 40px auto;
}
</style>