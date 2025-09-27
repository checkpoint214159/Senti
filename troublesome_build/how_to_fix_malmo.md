# The problem

On the first run of the line:

obs = env.reset(),

minedojo will do some funny install of the minecraft environment using Gradle and Malmo. I have zero clue what that means at all. However, it got in the way for a few hours.
That didn't stop me from hunting down the fix.

# The fix

The core issue is that all of the websites listen in the build.gradle config of Malmo, may not be alive for whatever reason.
When those locations are searched for, nothing can be found, so Gradle oofs and dies.
This is horrible! To fix, have the required file that its looking for already in local storage, and point to it in the gradle config.
This is at least what I think is happening; people who actually know their stuff please feel free to correct me.
Correct or not, this works :D

For me (ben), since my project directory structure is already pre-determined, just run the command in the neighbouring txt file to fix everything, in case you got dementia and forgot how to resolve this.

1. Run

git clone https://github.com/verityw/MixinGradle-dcfaf61 ~/workspace/MixinGradle-dcfaf61

~/workspace being some temporary directory, or a filepath on your local storage. Make a subdirectory MixinGradle-dcfaf61; this was needed for me, and will be explained in the next step.

2. Modify build.gradle

There should be a copy of the file here, so feel free to edit it. Replace the following:

buildscript {
    repositories {

        maven { url 'https://jitpack.io' }
        jcenter()
        mavenCentral()
        maven {
            name = "forge"
            url = "https://maven.minecraftforge.net/"
        }
        maven {
            name = "sonatype"
            url = "https://oss.sonatype.org/content/repositories/snapshots/"
        }
    }
    dependencies {
        classpath 'org.ow2.asm:asm:6.0'
        classpath('com.github.SpongePowered:MixinGradle:dcfaf61'){ // 0.6
            // Because forgegradle requires 6.0 (not -debug-all) while mixinGradle depends on 5.0
            // and putting mixin here places it before forge in the class loader
            exclude group: 'org.ow2.asm', module: 'asm-debug-all'
        }

        classpath 'com.github.yunfanjiang:ForgeGradle:FG_2.2_patched-SNAPSHOT'
    }
}

with this:

buildscript {
    repositories {

        maven { url = 'https://jitpack.io' }
        jcenter()
        mavenCentral()
        maven {
            url ="file:~/workspace/MixinGradle-dcfaf61" // Local directory where the repository was cloned
        }
        maven {
            name = "forge"
            url = "https://maven.minecraftforge.net/"
        }
        maven {
            name = "sonatype"
            url = "https://oss.sonatype.org/content/repositories/snapshots/"
        }
    }
    dependencies {
        classpath 'org.ow2.asm:asm:6.0'
        // classpath('com.github.SpongePowered:MixinGradle:dcfaf61'){ // 0.6
        //     // Because forgegradle requires 6.0 (not -debug-all) while mixinGradle depends on 5.0
        //     // and putting mixin here places it before forge in the class loader
        //     exclude group: 'org.ow2.asm', module: 'asm-debug-all'
        // }
        classpath('MixinGradle-dcfaf61:MixinGradle:dcfaf61'){ // 0.6
            // Because forgegradle requires 6.0 (not -debug-all) while mixinGradle depends on 5.0
            // and putting mixin here places it before forge in the class loader
            exclude group: 'org.ow2.asm', module: 'asm-debug-all'
        }

        classpath 'com.github.brandonhoughton:ForgeGradle:FG_2.2_patched-SNAPSHOT'
    }
}

#### NOTE: The most crucial part here is that line:

maven {
            url = "file:~/workspace/MixinGradle-dcfaf61" // Local directory where the repository was cloned
        }

Also, if you found a very similar solution to this from [here](https://github.com/qiwang067/LS-Imagine/blob/main/docs/minedojo_installation.md), you might see that I've added a subdirectory 'MixinGradle-dcfaf61' to ~/workspace/. This was based on my experience, where in the error logs, I found:

- file:/home/user/workspace/MixinGradle-dcfaf61/MixinGradle/dcfaf61/MixinGradle-dcfaf61.jar

to be the location that was searched. If you clone into ~/workspace/, you will still fail the above, since idk why, they by default append the subdirectory MixinGradle-dcfaf61 to whatever url you gave them. So, create that subdirectory and clone MixinGradle into it after.

`Remember to copy it to the build.gradle location that MineDojo looks for. For instance, if you have a venv, it should be somewhere in venv_name/lib/python3.10/site-packages/minedojo/sim/Malmo/Minecraft/build.gradle`

`or for conda ~/.conda/envs/minedojo/lib/python3.9/site-packages/minedojo/sim/Malmo/Minecraft/build.gradle`

Once you copy it there, when doing env.reset, it will 

3. Fixed!

The next time you call env.reset(), if there is no cached Minecraft environment, this whole nasty thing will run and compile it. Now, it shouldn't break and your hairline doesn't need to be yanked out, out of frustration!