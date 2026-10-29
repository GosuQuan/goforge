package main

import (
	"fmt"

	"example.com/sample-repo/pkg/calc"
)

func main() {
	fmt.Println("2 + 3 =", calc.Add(2, 3))
	fmt.Println("2 - 3 =", calc.Sub(2, 3))
	if q, err := calc.Divide(6, 3); err == nil {
		fmt.Println("6 / 3 =", q)
	}
}
