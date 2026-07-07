// Test mimicking the FIXED index.html structure
let moveForward = false;

function init() {
    const onKeyDown = function() {
        moveForward = true;
    };
    
    // Simulate key press
    onKeyDown();
    
    // Shadowing bug REMOVED.
}

function animate() {
    return moveForward;
}

try {
    init();
    if(animate() === true) {
        console.log("FIX CONFIRMED: animate() successfully sees moveForward as true.");
    } else {
        console.log("BUG PERSISTS.");
    }
} catch (e) {
    console.log("BUG PERSISTS via ReferenceError: " + e.message);
}
