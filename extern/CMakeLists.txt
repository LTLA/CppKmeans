include(FetchContent)

FetchContent_Declare(
  aarand
  GIT_REPOSITORY https://github.com/LTLA/aarand
  GIT_TAG afb49e269e02000373c55ccc982a4817be2b9d9d
)

FetchContent_MakeAvailable(aarand)

FetchContent_Declare(
  powerit 
  GIT_REPOSITORY https://github.com/LTLA/powerit
  GIT_TAG 8507271d961d6f601de166f5a8820e77269616d7
)

FetchContent_MakeAvailable(powerit)
