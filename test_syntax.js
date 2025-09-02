// Test file to check for syntax errors
console.log('Testing JavaScript syntax');

// Test basic function
function testFunction() {
    console.log('Function works');
    return 'success';
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