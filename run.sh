#!/bin/bash


pgrep ueye | xargs sudo kill
sudo /usr/bin/ueyeusbd
sudo python3 main.py
