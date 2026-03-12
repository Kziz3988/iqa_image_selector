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

    <div class="actions">
      <el-button type="primary" @click="submitUpload">
        上传图片
      </el-button>
    </div>

    <el-dialog v-model="dialogVisible">
      <img :src="dialogImageUrl" style="width:100%" />
    </el-dialog>
  </div>
</template>

<script setup>
import { ref } from "vue"
import axios from "axios"
import { Plus } from "@element-plus/icons-vue"
import { messageError, messageSuccess } from "@/utils/message"

const fileList = ref([])
const dialogVisible = ref(false)
const dialogImageUrl = ref("")
const API_URL = "http://localhost:8000/upload"

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
    messageError("只能上传图片格式文件")
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
    const res = await axios.post(API_URL, formData, {
      headers: {
        "Content-Type": "multipart/form-data"
      }
    })
    console.log(res.data)
    messageSuccess("上传成功")
  } catch (err) {
    console.error(err)
    messageError("上传失败")
  }
}
</script>

<style scoped>
.upload-page {
  width: 600px;
  margin: auto;
  padding-top: 40px;
}

.actions {
  margin-top: 20px;
}
</style>