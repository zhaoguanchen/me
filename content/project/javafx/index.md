---
title: Lab Resource Sharing System
summary: This is a Lab Resource Sharing System that was developed with JavaFX.
tags:
  - JavaFX
date: '2021-03-10T00:00:00Z'

# Optional external URL for project (replaces project detail page).
external_link: ''

image:
  caption:
  focal_point: Smart

links:
  - icon: github
    icon_pack: fab
    name: Follow
    url: https://github.com/zhaoguanchen/JavaFx-Lab-Resource-System

url_code: 'https://github.com/zhaoguanchen/JavaFx-Lab-Resource-System'
# url_pdf: ''
# url_slides: ''
url_video: 'https://youtu.be/U4hvyrmi904'


# Slides (optional).
#   Associate this project with Markdown slides.
#   Simply enter your slide deck's filename without extension.
#   E.g. `slides = "example-slides"` references `content/slides/example-slides.md`.
#   Otherwise, set `slides = ""`.
slides: ''
---

# JavaFX Project

This is a Lab Resource Sharing System that was developed with JavaFX.

## Tech Stack

**Framework**

We choose [JavaFX](https://openjfx.io/) to develop this desktop application. JavaFX is an open-source, next-generation client application platform for desktop, mobile, and embedded systems built on Java.

**Database**

MySQL (RDS on AWS)

## How to run

The recommand IDE is **IntelliJ IDEA**.

### 1. Set JDK 11

Go to File -> Project Structure -> Project, and set the project SDK to 11. You can also set the language level to 11.

### 2. Create a library

Go to File -> Project Structure -> Libraries and add the libs/ as a library to the project.

### 3. Add VM options

Go to Preferences (File -> Settings) -> Appearance & Behavior -> Path Variables, define the name of the variable as PATH_TO_FX, browse to the lib folder of the `/libs` to set its value, and click apply.  Also, you can download the latest version of `JavaFX` and redefine the variable path.

  

click on Run -> Edit Configurations... and add VM options:

```
--module-path ${PATH_TO_FX} --add-modules javafx.controls,javafx.fxml
```

### 4. Modify Database Connection

The path that saves the database information is `gczhao/database/DatabaseHandler.java`. Remember to replace these settings with your database information.

### 5. Execute SQL Statement

The script is located at `sql/script.sql`, which includes DDL and DML.

### 6. Run the project

Click Run -> Run... to run the project, now it should work fine.  

???					

## Demo

[Click](https://youtu.be/U4hvyrmi904) to watch the demo video.

## Other

email guanchenzhao@gmail.com for more information.

