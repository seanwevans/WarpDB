#include "arrow_utils.hpp"
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <stdexcept>

struct ArrowBufferInfo {
    void* data = nullptr;
    size_t size = 0;
    bool shared = false;
    int fd = -1;
    std::string name;
};

static void release_arrow_array(struct ArrowArray* array) {
    if (!array || !array->private_data) return;
    ArrowBufferInfo* info = static_cast<ArrowBufferInfo*>(array->private_data);
    if (info->shared) {
        munmap(info->data, info->size);
        if (info->fd >= 0) {
            close(info->fd);
            shm_unlink(info->name.c_str());
        }
    } else {
        free(info->data);
    }
    delete info;
    delete[] array->buffers;
    array->release = nullptr;
}

static void release_arrow_schema(struct ArrowSchema* schema) {
    schema->release = nullptr;
}

void export_to_arrow(const float* data, int64_t length, bool use_shared_memory,
                     ArrowArray* out_array, ArrowSchema* out_schema) {
    if (!out_array || !out_schema) throw std::invalid_argument("Null output");
    ArrowBufferInfo* info = new ArrowBufferInfo();
    info->size = sizeof(float) * length;
    info->shared = use_shared_memory;

    if (use_shared_memory) {
        info->name = "/warpdb_result";
        info->fd = shm_open(info->name.c_str(), O_CREAT | O_RDWR, 0600);
        if (info->fd < 0) {
            delete info;
            throw std::runtime_error("shm_open failed");
        }
        if (ftruncate(info->fd, info->size) != 0) {
            close(info->fd);
            delete info;
            throw std::runtime_error("ftruncate failed");
        }
        info->data = mmap(nullptr, info->size, PROT_READ | PROT_WRITE,
                          MAP_SHARED, info->fd, 0);
        if (info->data == MAP_FAILED) {
            close(info->fd);
            delete info;
            throw std::runtime_error("mmap failed");
        }
    } else {
        info->data = malloc(info->size);
        if (!info->data) {
            delete info;
            throw std::bad_alloc();
        }
    }
    std::memcpy(info->data, data, info->size);

    out_array->length = length;
    out_array->null_count = 0;
    out_array->offset = 0;
    out_array->n_buffers = 2;
    out_array->n_children = 0;
    out_array->buffers = new const void*[2];
    out_array->buffers[0] = nullptr; // no null bitmap
    out_array->buffers[1] = info->data;
    out_array->children = nullptr;
    out_array->dictionary = nullptr;
    out_array->release = release_arrow_array;
    out_array->private_data = info;

    out_schema->format = "f"; // float32
    out_schema->name = "result";
    out_schema->metadata = nullptr;
    out_schema->flags = ARROW_FLAG_NULLABLE;
    out_schema->n_children = 0;
    out_schema->children = nullptr;
    out_schema->dictionary = nullptr;
    out_schema->release = release_arrow_schema;
    out_schema->private_data = nullptr;
}
