macro(open3d_add_app_gui SRC_DIR APP_NAME TARGET_NAME)
    set(APPS_DIR "${PROJECT_SOURCE_DIR}/cpp/apps")
    set(SOURCE_DIR "${APPS_DIR}/${SRC_DIR}")

    if (APPLE)
        file(GLOB OBJC_FILES "${SOURCE_DIR}/*.mm")
        target_sources(${TARGET_NAME} PRIVATE ${OBJC_FILES})

        file(GLOB RESOURCE_FILES "${SOURCE_DIR}/*.icns")
        list(APPEND RESOURCE_FILES "${SOURCE_DIR}/Assets.car")

        set(INFO_PLIST "${SOURCE_DIR}/Info.plist.in")

        set(MACOSX_BUNDLE_NAME ${APP_NAME})
        set(MACOSX_BUNDLE_EXECUTABLE_NAME ${APP_NAME})
        set(MACOSX_BUNDLE_GUI_IDENTIFIER com.isl-org.open3d.${APP_NAME})
        set(MACOSX_BUNDLE_LONG_VERSION_STRING ${PROJECT_VERSION_THREE_NUMBER})
        set(MACOSX_BUNDLE_SHORT_VERSION_STRING ${PROJECT_VERSION_THREE_NUMBER})
        set(MACOSX_BUNDLE_BUNDLE_VERSION ${PROJECT_VERSION_THREE_NUMBER})
        set(MACOSX_BUNDLE_COPYRIGHT "Copyright (c) 2018-2025 www.open3d.org")
    endif()

    # Copy the resource files. This needs to be done as a post-build step
    # because on macOS, we don't know the bundle directory until build time
    # if we are using Xcode.
    set(RESOURCE_FILES ${GUI_RESOURCE_FILES} ${RESOURCE_FILES})
    if (APPLE)
        set(RESOURCE_DIR_NAME "Contents/Resources")
    else ()
        set(RESOURCE_DIR_NAME "resources")
    endif()

    # $<TARGET_BUNDLE_DIR> does not exist at config time, so we need to
    # duplicate the post build step on macOS and non-macOS
    if (APPLE)
        add_custom_command(TARGET "${TARGET_NAME}"
            POST_BUILD
            # copy the resource files into the bundle
            COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_BUNDLE_DIR:${TARGET_NAME}>/${RESOURCE_DIR_NAME}"
            COMMAND ${CMAKE_COMMAND} -E copy ${RESOURCE_FILES} "$<TARGET_BUNDLE_DIR:${TARGET_NAME}>/${RESOURCE_DIR_NAME}"
            # copy external libraries (e.g. SDL into the bundle and fixup
            # the search paths
            COMMAND ${APPS_DIR}/fixup_macosx_bundle.sh "$<TARGET_BUNDLE_DIR:${TARGET_NAME}>"
        )
    else ()
        set(APP_DIR "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
        add_custom_command(TARGET "${TARGET_NAME}"
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E make_directory "${APP_DIR}/${RESOURCE_DIR_NAME}"
            COMMAND ${CMAKE_COMMAND} -E copy ${RESOURCE_FILES} "${APP_DIR}/${RESOURCE_DIR_NAME}"
        )
        if (UNIX)
            install(DIRECTORY   "${APP_DIR}"
                    DESTINATION "${CMAKE_INSTALL_PREFIX}/bin"
                    USE_SOURCE_PERMISSIONS)
            if (CMAKE_INSTALL_PREFIX MATCHES "^(/usr/local|/opt)")
                set(DESKTOP_INSTALL_DIR "/usr/share" CACHE PATH "The install directory for the desktop apps")
            else()
                set(DESKTOP_INSTALL_DIR "$ENV{HOME}/.local/share" CACHE PATH "The install directory for the desktop apps")
            endif()
            configure_file("${SOURCE_DIR}/${TARGET_NAME}.desktop.in"
                           "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${APP_NAME}.desktop")
            # Install using freedesktop.org standards
            install(FILES "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${APP_NAME}.desktop"
                    DESTINATION "${DESKTOP_INSTALL_DIR}/applications")
            install(FILES "${SOURCE_DIR}/icon.svg"
                    DESTINATION "${DESKTOP_INSTALL_DIR}/icons/hicolor/scalable/apps"
                    RENAME "${APP_NAME}.svg")
            install(FILES "${SOURCE_DIR}/${TARGET_NAME}.xml"
                    DESTINATION "${DESKTOP_INSTALL_DIR}/mime/packages"
                    RENAME "${APP_NAME}.xml")
            # Various caches need to be updated for the app to become visible
            install(CODE "execute_process(COMMAND ${SOURCE_DIR}/postinstall-linux.sh)")
            configure_file("${SOURCE_DIR}/Debian/CMakeLists.in.txt"
                "${CMAKE_BINARY_DIR}/package-${TARGET_NAME}-deb/CMakeLists.txt" @ONLY)
            add_custom_target(package-${TARGET_NAME}-deb
                COMMAND cp -a "${CMAKE_BINARY_DIR}/${APP_NAME}" .
                COMMAND cp "${SOURCE_DIR}/icon.svg" "${APP_NAME}/${APP_NAME}.svg"
                COMMAND cp "${SOURCE_DIR}/${TARGET_NAME}.xml" "${APP_NAME}/"
                COMMAND cp "${SOURCE_DIR}/${TARGET_NAME}Launcher.sh" "${APP_NAME}/"
                COMMAND "${CMAKE_COMMAND}" -S .
                COMMAND "${CMAKE_COMMAND}" --build . -t package
                WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/package-${TARGET_NAME}-deb/"
                DEPENDS ${TARGET_NAME})
        elseif (WIN32)
            target_sources(${TARGET_NAME} PRIVATE "${SOURCE_DIR}/icon.rc")   # add icon

            # MSVC puts the binary in bin/Open3D/Release/Open3D.exe
            # so we can't just install() the build results, and need to do them piecemeal.
            install(DIRECTORY   "${APP_DIR}/resources"
                    DESTINATION "${CMAKE_INSTALL_PREFIX}/bin/${APP_NAME}"
                    USE_SOURCE_PERMISSIONS)
            install(FILES   "${APP_DIR}/$<CONFIG>/${TARGET_NAME}.exe"
                    DESTINATION "${CMAKE_INSTALL_PREFIX}/bin/${APP_NAME}"
                    RENAME "${APP_NAME}.exe")
        else()
            install(DIRECTORY   "${APP_DIR}"
                    DESTINATION "${CMAKE_INSTALL_PREFIX}/bin"
                    USE_SOURCE_PERMISSIONS)
        endif()
    endif()
endmacro()

macro(open3d_add_app_common SRC_DIR APP_NAME TARGET_NAME)
    set(APPS_DIR "${PROJECT_SOURCE_DIR}/cpp/apps")
    set(SOURCE_DIR "${APPS_DIR}/${SRC_DIR}")

    file(GLOB SOURCE_FILES "${SOURCE_DIR}/*.cpp")
    file(GLOB HEADER_FILES "${SOURCE_DIR}/*.h")

    if (APPLE)
        add_executable(${TARGET_NAME} ${SOURCE_FILES} ${HEADER_FILES})
        set_target_properties(${TARGET_NAME} PROPERTIES
                              MACOSX_BUNDLE TRUE
                              MACOSX_BUNDLE_INFO_PLIST "${INFO_PLIST}"
                              XCODE_ATTRIBUTE_CODE_SIGN_IDENTITY "" # disable
                              OUTPUT_NAME ${APP_NAME}
                              INSTALL_RPATH "@loader_path;@loader_path/../lib/")
    elseif (WIN32)
        set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/../${APP_NAME}")
        add_executable(${TARGET_NAME} ${SOURCE_FILES} ${HEADER_FILES})
    else()
        set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/../${APP_NAME}")
        add_executable(${TARGET_NAME} ${SOURCE_FILES} ${HEADER_FILES})
        set_target_properties(${TARGET_NAME} PROPERTIES
                            OUTPUT_NAME ${APP_NAME}
                            INSTALL_RPATH "$ORIGIN;$ORIGIN/../lib/")
    endif()

    target_link_libraries(${TARGET_NAME} PRIVATE Open3D::Open3D TBB::tbb ${ARGN})
    set_target_properties(${TARGET_NAME} PROPERTIES FOLDER "apps")

    if (BUILD_SYCL_MODULE)
        find_package(IntelSYCL REQUIRED)   # requires cmake>=3.25 on Windows
        add_sycl_to_target(TARGET ${TARGET_NAME})
    endif()

    open3d_link_3rdparty_libraries(${TARGET_NAME})
    open3d_show_and_abort_on_warning(${TARGET_NAME})
    open3d_set_global_properties(${TARGET_NAME})
endmacro()

if (BUILD_GUI)
    open3d_add_app_common(Open3DViewer Open3D Open3DViewer)
    open3d_add_app_gui(Open3DViewer Open3D Open3DViewer)
endif()

open3d_add_app_common(OfflineReconstruction OfflineReconstruction OfflineReconstruction)
