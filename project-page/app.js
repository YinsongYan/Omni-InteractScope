const examples = document.querySelectorAll('.example-item');

for (const example of examples) {
  example.addEventListener('toggle', () => {
    if (example.open) {
      for (const other of examples) {
        if (other !== example) other.open = false;
      }
    } else {
      example.querySelector('video')?.pause();
    }
  });
}
