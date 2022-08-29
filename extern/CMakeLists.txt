include(FetchContent)

FetchContent_Declare(
  aarand
  GIT_REPOSITORY https://github.com/LTLA/aarand
  GIT_TAG f61f6b08453cfa8d2681f32b795626f8cd21c724
)

FetchContent_MakeAvailable(aarand)

FetchContent_Declare(
  powerit 
  GIT_REPOSITORY https://github.com/LTLA/powerit
  GIT_TAG 5fbbe4a5fb98e75d673e47ac94a2c828f3781271
)

FetchContent_MakeAvailable(powerit)
