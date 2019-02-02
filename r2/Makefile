R2_PLUGIN_PATH=$(shell r2 -hh|grep USER_PLUGINS|awk '{print $$2}')
SO_EXT=$(shell uname|grep -q Darwin && echo dylib || echo so)

# RAsm plugin
RASM_NAME=hexagon_asm
RASM_CFLAGS=-g -fPIC $(shell pkg-config --cflags r_asm)
RASM_LDFLAGS=-g -fPIC -shared $(shell pkg-config --libs r_asm)
RASM_OBJS=asm_hexagon.o hexagon.o hexagon_disas.o
RASM_SRC=$(pathsubst %.o, %.c, $(RASM_OBJS))
RASM_LIB=$(RASM_NAME).$(SO_EXT)

# RAnal plugin
RANAL_NAME=hexagon_anal
RANAL_CFLAGS=-g -fPIC $(shell pkg-config --cflags r_anal)
RANAL_LDFLAGS=-g -fPIC -shared $(shell pkg-config --libs r_anal)
RANAL_OBJS=anal_hexagon.o hexagon.o hexagon_disas.o hexagon_anal.o
RANAL_SRC=$(pathsubst %.o, %.c, $(RASM_OBJS))
RANAL_LIB=$(RANAL_NAME).$(SO_EXT)

all: $(RASM_LIB) $(RANAL_LIB)

clean:
	rm -f $(RASM_LIB) $(RANAL_LIB) $(RASM_OBJS) $(RANAL_OBJS)

$(RASM_LIB): $(RASM_OBJS)
	$(CC) $(RASM_CFLAGS) $(RASM_LDFLAGS) $(RASM_OBJS) -o $(RASM_LIB)

$(RANAL_LIB): $(RANAL_OBJS)
	$(CC) $(RANAL_CFLAGS) $(RANAL_LDFLAGS) $(RANAL_OBJS) -o $(RANAL_LIB)

%.o: %.c
	$(CC) $(RASM_CFLAGS) $(RANAL_CFLAGS) -c $< -o $@

install:
	cp -f $(RASM_NAME).$(SO_EXT) $(R2_PLUGIN_PATH)
	cp -f $(RANAL_NAME).$(SO_EXT) $(R2_PLUGIN_PATH)

uninstall:
	rm -f $(R2_PLUGIN_PATH)/$(RASM_NAME).$(SO_EXT)
	rm -f $(R2_PLUGIN_PATH)/$(RANAL_NAME).$(SO_EXT)

