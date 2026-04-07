#!/usr/bin/env bash
set -e

MOCAP=/home/harri/projects/mocap_videos/20251115-aikido
OUT=/tmp/led_rois

mkdir -p "$OUT"

# cam0 ace2pro 23x25
ffmpeg -hide_banner -loglevel error -y \
  -i "$MOCAP/20251115-aikido-harri-timo-ace2pro-mega-120fps.mp4" \
  -vf "crop=23:25:1455:1743,format=gray" \
  -f rawvideo -pix_fmt gray "$OUT/cam0_ace2pro_23x25.raw"

# cam1 gopromini-01 26x22
ffmpeg -hide_banner -loglevel error -y \
  -i "$MOCAP/20251115-aikido-harri-timo-gopromini-01-120fps.MP4" \
  -vf "crop=26:22:2885:1325,format=gray" \
  -f rawvideo -pix_fmt gray "$OUT/cam1_gopromini01_26x22.raw"

# cam2 gopromini-02 21x19
ffmpeg -hide_banner -loglevel error -y \
  -i "$MOCAP/20251115-aikido-harri-timo-gopromini-02-120fps.MP4" \
  -vf "crop=21:19:914:1333,format=gray" \
  -f rawvideo -pix_fmt gray "$OUT/cam2_gopromini02_21x19.raw"

# cam3 instax3 29x26
ffmpeg -hide_banner -loglevel error -y \
  -i "$MOCAP/20251115-aikido-harri-timo-instax3-60fps.mp4" \
  -vf "crop=29:26:2524:1426,format=gray" \
  -f rawvideo -pix_fmt gray "$OUT/cam3_instax3_29x26.raw"

# cam4 pixel9 11x12
ffmpeg -hide_banner -loglevel error -y \
  -i "$MOCAP/20251115-aikido-harri-timo-pixel9-wideangle-120fps.mp4" \
  -vf "crop=11:12:775:537,format=gray" \
  -f rawvideo -pix_fmt gray "$OUT/cam4_pixel9_11x12.raw"

# cam5 r5 15x15
ffmpeg -hide_banner -loglevel error -y \
  -i "$MOCAP/20251115-aikido-harri-timo-r5-17-40-zoom-17-60fps.MP4" \
  -vf "crop=15:15:2545:1112,format=gray" \
  -f rawvideo -pix_fmt gray "$OUT/cam5_r5_15x15.raw"

echo "Done. Playback commands:"
echo "  ffplay -f rawvideo -video_size 23x25 -pix_fmt gray -vf scale=460:500:flags=neighbor $OUT/cam0_ace2pro_23x25.raw"
echo "  ffplay -f rawvideo -video_size 26x22 -pix_fmt gray -vf scale=520:440:flags=neighbor $OUT/cam1_gopromini01_26x22.raw"
echo "  ffplay -f rawvideo -video_size 21x19 -pix_fmt gray -vf scale=420:380:flags=neighbor $OUT/cam2_gopromini02_21x19.raw"
echo "  ffplay -f rawvideo -video_size 29x26 -pix_fmt gray -vf scale=580:520:flags=neighbor $OUT/cam3_instax3_29x26.raw"
echo "  ffplay -f rawvideo -video_size 11x12 -pix_fmt gray -vf scale=440:480:flags=neighbor $OUT/cam4_pixel9_11x12.raw"
echo "  ffplay -f rawvideo -video_size 15x15 -pix_fmt gray -vf scale=450:450:flags=neighbor $OUT/cam5_r5_15x15.raw"
