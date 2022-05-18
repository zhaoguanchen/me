---
title: Application of Two-Pointer to Array and Linked List Problems
subtitle: When dealing with problems related to arrays and linked lists, Two-Pointer techniques are often used.

# Summary for listings and search engines
summary: When dealing with problems related to arrays and linked lists, Two-Pointer techniques are often used.

# Link this post with a project
projects: []

# Date published
date: '2022-03-13T00:00:00Z'

# Date updated
lastmod: '2020-05-13T00:00:00Z'

# Is this an unpublished draft?
draft: false

# Show this page in the Featured widget?
featured: false

# Featured image
# Place an image named `featured.jpg/png` in this page's folder and customize its options here.
image:
  caption: ''
  focal_point: ''
  placement: 2
  preview_only: false

authors:
  - Guanchen Zhao
tags:
  - Two-Pointer

categories:
  - Algorithm
---

## Overview

When dealing with problems related to arrays and linked lists, Two-Pointer techniques are often used. Two-Pointer techniques are mainly divided into two categories: left and right pointers and fast and slow pointers.
Left and right pointers are two pointers that move towards each other or opposite, and fast and slow pointers are two pointers that move in the same direction, one is fast and the other is slow.
For singly linked list problems, most of the techniques we use belong to fast and slow pointers, such as linked list ring judgment, penultimate K-th linked list node, and so on. They all complete tasks through a fast pointer and a slow pointer.
In an array, although there is no real pointer, we can treat the index as a pointer in the array, so that we can also perform the double-pointer trick in the array.
This article is divided into two parts, using the Two-Pointer technique to solve the linked list problem and solve the array problem.
