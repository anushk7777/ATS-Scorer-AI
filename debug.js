// Debug script to test JavaScript syntax
console.log('Debug script loaded');

// Test basic function
function testFunction() {
    console.log('Test function works');
    return true;
}

// Test async function
async function testAsync() {
    console.log('Async function works');
    return 'success';
}

// Test conditional return
function testConditionalReturn(condition) {
    if (condition) {
        return 'condition met';
    }
    return 'condition not met';
}

// Run tests
testFunction();
testAsync();
testConditionalReturn(true);
console.log('All tests completed');