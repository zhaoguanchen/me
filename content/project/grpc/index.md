---
title: A gRPC project based on AES cryptosystem.
summary: A gRPC project based on AES cryptosystem.
tags:
  - gRPC
date: '2021-03-10T00:00:00Z'

# Optional external URL for project (replaces project detail page).
external_link: ''

reading_time: true  # Show estimated reading time?
share: true  # Show social sharing links?
profile: true  # Show author profile?
commentable: true  # Allow visitors to comment? Supported by the Page, Post, and Docs content types.
editable: false  # Allow visitors to edit the page? Supported by the Page, Post, and Docs content types.

# Featured image
# To use, place an image named `featured.jpg/png` in your page's folder.
# Placement options: 1 = Full column width, 2 = Out-set, 3 = Screen-width
# Focal point options: Smart, Center, TopLeft, Top, TopRight, Left, Right, BottomLeft, Bottom, BottomRight
# Set `preview_only` to `true` to just use the image for thumbnails.
image:
  placement:
  caption:
  focal_point: Smart
  preview_only: false
  alt_text: gRPC.

# assets/media/header.png
header:
  image: "icon.png"
  caption:

draft: false


links:
  - icon: github
    icon_pack: fab
    name: Follow
    url: https://github.com/zhaoguanchen/grpc-cipher

url_code: 'https://github.com/zhaoguanchen/grpc-cipher'
# url_pdf: ''
# url_slides: ''
url_video: 'https://youtu.be/Yl7gV_KAfGk'


# Slides (optional).
#   Associate this project with Markdown slides.
#   Simply enter your slide deck's filename without extension.
#   E.g. `slides = "example-slides"` references `content/slides/example-slides.md`.
#   Otherwise, set `slides = ""`.
slides: ''
---

# gRPC Project

## What is gRPC?

In gRPC, a client application can directly call a method on a server application on a different machine as if it were a local object, making it easier for you to create distributed applications and services. As in many RPC systems, gRPC is based around the idea of defining a service, specifying the methods that can be called remotely with their parameters and return types. On the server side, the server implements this interface and runs a gRPC server to handle client calls. On the client side, the client has a stub (referred to as just a client in some languages) that provides the same methods as the server.



![Concept Diagram](https://grpc.io/img/landing-2.svg)



gRPC clients and servers can run and talk to each other in a variety of environments - from servers inside Google to your own desktop - and can be written in any of gRPC’s supported languages. So, for example, you can easily create a gRPC server in Java with clients in Go, Python, or Ruby. In addition, the latest Google APIs will have gRPC versions of their interfaces, letting you easily build Google functionality into your applications.

By default, gRPC uses [Protocol Buffers](https://developers.google.com/protocol-buffers/docs/overview), Google’s mature open source mechanism for serializing structured data (although it can be used with other data formats such as JSON).

## What is this project?

The function of this project is RSA encryption. We established a server with encryption and decryption function, and created relevant clients to call.

This project consists of two parts: server (under `/RPC` directory) and client (under `/client` directory).

## How to Run?

The base IDE is **Eclipse**.

### 1. Start Server

The start function is located at `rpc/src/main/java/GRCPServer.java`. click `Run as Java Application`.

### 2. Start Client

The start function is located at `client/src/main/java/client/GrpcClient.java`. click `Run as Java Application`.

Then you will see the parameters and results in the console window.

## Demo

Please click [HERE](https://youtu.be/Yl7gV_KAfGk) to watch the demo video.

## Other

email guanchenzhao@gmail.com for more information.







