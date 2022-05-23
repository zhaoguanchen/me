---
title: What is Monotone Stack?
subtitle: A monotonic stack is a stack whose elements are monotonically
  increasing or descreasing.
date: 2022-03-02T00:00:00.000Z
summary: ""
draft: false
featured: false
authors: []
lastmod: 2020-12-13T00:00:00Z
tags:
  - Monotonic Stack
categories:
  - Algorithm
projects: []
image:
  caption: ""
  focal_point: ""
  placement: 2
  preview_only: false
---
Stack is a very simple data structure. His logical order of first in and last out conforms to the characteristics of some problems, such as function call stack.
Monotone stack is actually a stack. It just uses some ingenious logic to keep the elements in the stack orderly (increasing or decreasing) every time a new element is put into the stack.
Monotone stack is not widely used. It only deals with a typical problem called next greater element.
There are a total of n elements. Each element is pushed into the stack once, and will be pop once at most, without any redundant operations. Therefore, the total calculation scale is directly proportional to the element scale n, that is, the complexity of O (n).

Click [here](https://gczhao.cn/Leetcode/DataStruct/MonotoneStack/) for more information.