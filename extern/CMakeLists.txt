include(FetchContent)

FetchContent_Declare(
  aarand
  GIT_REPOSITORY https://github.com/LTLA/aarand
  GIT_TAG master # ^1.0.0
)

FetchContent_Declare(
  powerit 
  GIT_REPOSITORY https://github.com/LTLA/powerit
  GIT_TAG master # ^2.0.0
)

FetchContent_Declare(
  subpar 
  GIT_REPOSITORY https://github.com/LTLA/subpar
  GIT_TAG master # ^0.3.0
)

FetchContent_MakeAvailable(aarand)
FetchContent_MakeAvailable(powerit)
FetchContent_MakeAvailable(subpar)
