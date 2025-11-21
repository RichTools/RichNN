set -xe

gcc -Wall -o iris ./examples/iris.c

FLAG=$1

# Check if a flag argument is provided
if [[ -n "$FLAG" ]]; then
  # Handle the flag case (e.g., '-Log')
  if [[ "$1" == '-Log' ]]; then
    ./iris "$@"  # Pass all arguments to xor (including '-Log')
  else
    echo "Invalid flag: '$1'" >&2
    exit 1
  fi
else
  # No flag provided, call xor without arguments
  ./iris
fi
