include(FetchContent)

FetchContent_Declare(
  aarand
  GIT_REPOSITORY https://github.com/LTLA/aarand
  GIT_TAG master
)

FetchContent_MakeAvailable(aarand)

FetchContent_Declare(
  powerit 
  GIT_REPOSITORY https://github.com/LTLA/powerit
  GIT_TAG master
)

FetchContent_MakeAvailable(powerit)
