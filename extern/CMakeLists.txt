include(FetchContent)

FetchContent_Declare(
  aarand
  GIT_REPOSITORY https://github.com/LTLA/aarand
  GIT_TAG 84d48b65d49ce8b844398f11aff3015b86e17197
)

FetchContent_MakeAvailable(aarand)

FetchContent_Declare(
  powerit 
  GIT_REPOSITORY https://github.com/LTLA/powerit
  GIT_TAG 705c4a6209baeaf4a246c8a61c46ecd0d3473511
)

FetchContent_MakeAvailable(powerit)
