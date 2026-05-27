<template>
  <div class="upload-page">
    <div class="upload-header">
      <h2>{{ isProcessing ? "正在进行优选..." : "请上传待优选的图像集" }}</h2>

      <div class="model-select" v-if="!isProcessing">
        <span>选择模型：</span>

        <el-select v-model="selectedModel" style="width: 200px">
          <el-option label="自动选择" value="Selector"/>
          <el-option label="ARNIQA" value="ARNIQA"/>
          <el-option label="MANIQA" value="MANIQA"/>
          <el-option label="DBCNN" value="DBCNN"/>
        </el-select>
      </div>

      <div
        class="actions"
        v-if="!isProcessing"
      >
        <el-button
          type="danger"
          v-if="fileList.length > 0"
          @click="clearImages"
        >
          清空图像
        </el-button>

        <el-button
          type="primary"
          @click="submitUpload"
        >
          上传图像
        </el-button>
      </div>
    </div>

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
    >
      <el-icon>
        <Plus />
      </el-icon>
    </el-upload>

    <el-dialog v-model="dialogVisible">
      <img
        :style="{ width: '100%' }"
        :src="dialogImageUrl"
      />
    </el-dialog>

    <div
      class="progress"
      v-if="isProcessing"
    >
      <el-icon class="loading-icon">
        <Loading />
      </el-icon>

      <span class="progress-text">
        {{ progress }}
      </span>

      <el-progress
        v-if="progressPercent !== null"
        :percentage="progressPercent"
        stroke-width="10"
        :text-inside="false"
        style="width: 80%; margin-top: 10px;"
      />
    </div>
  </div>
</template>

<script setup>
import { ref, defineEmits, inject } from "vue"
import axios from "axios"
import { Plus, Loading } from "@element-plus/icons-vue"
import { genFileId } from "element-plus"
import { messageError, messageSuccess } from "@/utils/message"
import { v4 as uuidv4 } from "uuid"

const emit = defineEmits(["upload-success"])

const UPLOAD_URL = "http://localhost:8000/upload"
const PROCESS_URL = "http://localhost:8000/process"

const fileList = ref([])

const dialogVisible = ref(false)
const dialogImageUrl = ref("")

const progress = ref("")
const progressPercent = ref(null)

const ws = ref(null)

const selectedModel = ref("Selector")

const isProcessing = inject("isProcessing")

const fileStorage = ref(new Map())

const isValidImageFile = (file) => {
  const target = file.raw || file
  if (
    target.type &&
    target.type.startsWith("image/")
  ) {
    return true
  }

  return false
}

const handleChange = (uploadFile, uploadFiles) => {
  if (isProcessing.value) {
    messageError("任务进行中，无法添加图像")
    return
  }

  const filteredFiles = []

  const nameSet = new Set()

  for (const file of uploadFiles) {
    const raw = file.raw || file

    if (!isValidImageFile(raw)) {
      messageError(`"${file.name}" 不是合法图片文件`)
      continue
    }

    if (nameSet.has(file.name)) {
      messageError(`"${file.name}" 已存在`)
      continue
    }

    nameSet.add(file.name)

    if (!file.uid) {
      file.uid = genFileId()
    }

    filteredFiles.push(file)
  }

  fileList.value = filteredFiles
}

const handleRemove = (file, files) => {
  if (isProcessing.value) {
    messageError("任务进行中，无法删除图像")
    return
  }

  if (file.uuid) {
    fileStorage.value.delete(file.uuid)
  }

  fileList.value = files
}

const handlePreview = (file) => {

  if (file.url) {
    dialogImageUrl.value = file.url
  } else {
    dialogImageUrl.value = URL.createObjectURL(file.raw)
  }

  dialogVisible.value = true
}

const submitUpload = async () => {
  if (fileList.value.length === 0) {
    messageError("请先选择要上传的图像")
    return
  }

  const formData = new FormData()

  fileStorage.value.clear()

  fileList.value.forEach((item) => {
    const uuid = uuidv4()

    const extension = item.name.substring(
      item.name.lastIndexOf(".")
    )

    const newFileName = uuid + extension

    fileStorage.value.set(uuid, {
      originalName: item.name,
      blob: item.raw,
      uuid
    })

    item.uuid = uuid
    item.storageKey = newFileName

    formData.append(
      "files",
      item.raw,
      newFileName
    )
  })

  try {
    isProcessing.value = true

    progress.value = "任务启动中..."

    const uploadRes = await axios.post(
      UPLOAD_URL,
      formData
    )

    const task_id = uploadRes.data.task_id

    connectWS(task_id)

    await new Promise((resolve) => {
      ws.value.onopen = resolve
    })

    const processRes = await axios.get(
      `${PROCESS_URL}/${task_id}`,
      {
        params: {
          iqa_model: selectedModel.value
        }
      }
    )

    const data = processRes.data

    const resultData = {
      ...data,
      fileStorage: fileStorage.value
    }

    emit("upload-success", resultData)

    progress.value = "处理完毕"

    messageSuccess("处理完毕")
  } catch (err) {
    console.error(err)

    messageError(
      err.response?.data?.detail ||
      "上传或处理失败"
    )
  } finally {
    isProcessing.value = false
  }
}

const clearImages = () => {
  if (isProcessing.value) {
    messageError("任务进行中，无法清空图像")
    return
  }
  
  fileList.value = []
  fileStorage.value.clear()
}

const connectWS = (task_id) => {
  ws.value = new WebSocket(
    `ws://localhost:8000/ws/${task_id}`
  )

  ws.value.onmessage = (event) => {
    const data = JSON.parse(event.data)

    if (data.progress !== undefined) {
      progressPercent.value = Math.round(
        data.progress * 100
      )

      progress.value = data.message
    } else {
      progressPercent.value = null

      progress.value = data.message
    }
  }

  ws.value.onclose = () => {
    console.log("WebSocket closed")
  }
}
</script>

<style scoped>
.upload-page {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: flex-start;
  gap: 20px;
}

.upload-header {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 20px;
  flex-wrap: wrap;
  margin-bottom: 20px;
}

.upload-header h2 {
  margin: 0;
}

.model-select {
  display: flex;
  align-items: center;
  gap: 10px;
}

.actions {
  display: flex;
  gap: 10px;
}

.upload-image
  ::v-deep(.el-upload-list--picture-card) {
  flex-wrap: wrap;
  justify-content: center;
  padding: 0;
  margin: 0;
  list-style: none;
}

/* 只在任务进行时显示禁用样式 */
.upload-page:has(.progress) ::v-deep(.el-upload--picture-card),
.upload-page:has(.progress) ::v-deep(.el-upload-list__item) {
  cursor: not-allowed;
  opacity: 0.6;
  pointer-events: none;
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

.progress-text {
  color: #409EFF;
}

@keyframes rotating {
  from {
    transform: rotate(0deg);
  }

  to {
    transform: rotate(360deg);
  }
}
</style>