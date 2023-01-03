// RUN: %c-to-llvm %s | %apply-cucorr -S 2>&1 | %filecheck %s

// CHECK-NOT: Error

int main(void) { return 0; }
