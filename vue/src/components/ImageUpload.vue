<template>
  <div class="upload-page">
    <h2>请上传待优选的图像集</h2>

    <el-upload
      class="upload-image"
      drag
      multiple
      list-type="picture-card"
      accept="image/*"
      :auto-upload="false"
      :file-list="fileList"
      :on-change="handleChange"
      :on-remove="handleRemove"
      :on-preview="handlePreview"
      :before-upload="beforeUpload"
    >
      <el-icon><Plus /></el-icon>
    </el-upload>

    <div class="model-select" v-if="!isProcessing">
      <span>选择模型：</span>
      <el-select v-model="selectedModel" style="width: 200px">
        <el-option label="ARNIQA" value="ARNIQA" />
        <el-option label="MANIQA" value="MANIQA" />
        <el-option label="VCRNet" value="VCRNet" />
        <el-option label="DBCNN" value="DBCNN" />
      </el-select>
    </div>

    <div class="actions" v-if="!isProcessing">
      <el-button type="danger" v-if="fileList.length > 0" @click="clearImages">
        清空图像
      </el-button>
      <el-button type="primary" @click="submitUpload">
        上传图像
      </el-button>
    </div>

    <el-dialog v-model="dialogVisible">
      <img :src="dialogImageUrl" style="width:100%" />
    </el-dialog>

    <div class="progress" v-if="isProcessing">
      <el-icon class="loading-icon">
        <Loading />
      </el-icon>
      <span class="progress-text">{{ progress }}</span>
    </div>
  </div>
</template>

<script setup>
import { ref, defineEmits, inject } from "vue"
import axios from "axios"
import { Plus, Loading } from "@element-plus/icons-vue"
import { messageError, messageSuccess } from "@/utils/message"

const fileList = ref([])
const dialogVisible = ref(false)
const dialogImageUrl = ref("")
const UPLOAD_URL = "http://localhost:8000/upload"
const PROCESS_URL = "http://localhost:8000/process"
const emit = defineEmits(["upload-success"])
const progress = ref("")
const ws = ref(null)
const isProcessing = inject('isProcessing')
const selectedModel = ref("ARNIQA")

const handleChange = (file, files) => {
  fileList.value = files
}

const handleRemove = (file, files) => {
  fileList.value = files
}

const handlePreview = (file) => {
  dialogImageUrl.value = file.url || URL.createObjectURL(file.raw)
  dialogVisible.value = true
}

const beforeUpload = (file) => {
  const isImage = file.type.startsWith("image/")
  if (!isImage) {
    messageError("只能上传图像格式文件")
    return false
  }
  return true
}

const submitUpload = async () => {
  if (fileList.value.length === 0) return

  const formData = new FormData()
  fileList.value.forEach((item) => {
    formData.append("files", item.raw)
  })

  try {
    isProcessing.value = true
    progress.value = "任务启动中..."
    const uploadRes = await axios.post(UPLOAD_URL, formData)
    const task_id = uploadRes.data.task_id
    connectWS(task_id)
    await new Promise(resolve => {
      ws.value.onopen = resolve
    })

    const processRes = await axios.get(`${PROCESS_URL}/${task_id}`, {
      params: {
        iqa_model: selectedModel.value
      }
    })
    const data = processRes.data
    const fileMap = {}
    fileList.value.forEach(f => {
      fileMap[f.name] = f.raw
    })
    emit("upload-success", {
      ...data,
      fileMap
    })
    progress.value = "处理完毕"
    messageSuccess("处理完毕")
  } catch (err) {
    console.error(err)
    messageError("上传失败")
  } finally {
    isProcessing.value = false
  }
}

const clearImages = () => {
  fileList.value = []
}

const connectWS = (task_id) => {
  ws.value = new WebSocket(`ws://localhost:8000/ws/${task_id}`)
  ws.value.onmessage = (event) => {
    const data = JSON.parse(event.data)
    progress.value = data.message
  }

  ws.value.onclose = () => {
    console.log("WebSocket closed")
  }
}
</script>

<style scoped>
.upload-page {
  width: 600px;
  margin: auto;
  padding-top: 40px;
}

.upload-image ::v-deep(.el-upload-list--picture-card) {
  flex-wrap: wrap;
  justify-content: center;
  padding: 0;
  margin: 0;
  list-style: none;
}

.upload-image ::v-deep(.el-upload-list__item) {
  float: none !important;
  margin: 5px;
}

.actions {
  margin-top: 20px;
}

.model-select {
  margin: 20px 0;
  display: flex;
  justify-content: center;
  align-items: center;
  gap: 10px;
}

.progress {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  margin-top: 40px;
  font-size: 18px;
}

.loading-icon {
  margin-right: 10px;
  font-size: 20px;
  animation: rotating 1s linear infinite;
}

@keyframes rotating {
  from {
    transform: rotate(0deg);
  }
  to {
    transform: rotate(360deg);
  }
}

.progress-text {
  color: #409EFF;
}
</style>