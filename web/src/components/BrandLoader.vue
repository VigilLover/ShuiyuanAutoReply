<script setup lang="ts">
import mark from '../assets/brand-mark.svg?raw'
import smallMark from '../assets/brand-small.svg?raw'

withDefaults(defineProps<{ compact?: boolean }>(), { compact: false })
</script>

<template>
  <!-- Decorative: the containing status supplies the accessible loading text. -->
  <span class="brand-loader" :class="{ 'brand-loader--compact': compact }" aria-hidden="true" v-html="compact ? smallMark : mark"></span>
</template>

<style scoped>
.brand-loader { display: inline-flex; width: 88px; height: 90px; flex: none; color: var(--blue); }
.brand-loader :deep(svg) { display: block; width: 100%; height: 100%; color: inherit; overflow: visible; }
.brand-loader :deep(.brand-orbit) { animation: brand-breathe 2.8s ease-in-out infinite; transform-origin: center; }
.brand-loader :deep(.brand-eyes) { animation: brand-blink 5.6s ease-in-out infinite; transform-box: fill-box; transform-origin: center; }
.brand-loader :deep(.brand-dots circle) { animation: brand-message 1.4s ease-in-out infinite; }
.brand-loader :deep(.brand-dots circle:nth-child(2)) { animation-delay: .16s; }
.brand-loader :deep(.brand-dots circle:nth-child(3)) { animation-delay: .32s; }
.brand-loader--compact { width: 26px; height: 28px; }
.brand-loader--compact :deep(.brand-eyes) { animation: none; }
@keyframes brand-breathe { 0%, 100% { opacity: .4; transform: scale(.97); } 50% { opacity: .85; transform: scale(1); } }
@keyframes brand-message { 0%, 65%, 100% { opacity: .35; transform: translateY(0); } 30% { opacity: 1; transform: translateY(-2px); } }
@keyframes brand-blink { 0%, 43%, 49%, 100% { transform: scaleY(1); } 46% { transform: scaleY(.15); } }
@media (prefers-reduced-motion: reduce) {
  .brand-loader :deep(.brand-orbit), .brand-loader :deep(.brand-eyes), .brand-loader :deep(.brand-dots circle) { animation: none; }
}
</style>
