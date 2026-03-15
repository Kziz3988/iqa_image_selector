<template>
  <ImageUpload @upload-success="handleResult"/>
  <ImageResult v-if="result" :resultData="result"/>
</template>

<script setup>
import { ref, provide } from "vue"
import ImageUpload from './components/ImageUpload.vue'
import ImageResult from './components/ImageResult.vue'

const isProcessing = ref(false)
const result = ref(null)
const handleResult = (data) => {
  result.value = data
}

provide('isProcessing', isProcessing)
</script>

<style>
#app {
  font-family: Avenir, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #2c3e50;
  margin-top: 60px;
}
</style>
