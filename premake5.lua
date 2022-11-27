-- premake5.lua
workspace "RTTrace"
   architecture "x64"
   configurations { "Debug", "Release", "Dist" }
   startproject "RTTrace"

outputdir = "%{cfg.buildcfg}-%{cfg.system}-%{cfg.architecture}"
include "Walnut/WalnutExternal.lua"

include "RTTrace"