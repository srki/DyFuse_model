FetchContent_Declare(
        otx
        GIT_REPOSITORY https://github.com/srki/otx.git
)
FetchContent_MakeAvailable(otx)

FetchContent_Declare(
        mio
        GIT_REPOSITORY https://github.com/srki/mio.git
)
FetchContent_MakeAvailable(mio)

#add_subdirectory(betweenness-centrality)
#add_subdirectory(examples)
#add_subdirectory(simple)
#add_subdirectory(tricount)
add_subdirectory(stats)
#add_subdirectory(scheduler)
#add_subdirectory(bfs)

add_subdirectory(simple_model)
